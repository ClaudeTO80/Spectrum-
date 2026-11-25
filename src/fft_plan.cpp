#include "fft/fft_plan.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#include "fft/exceptions.hpp"

namespace fft {

namespace {

/**
 * @brief Returns true if @p n is a power of four greater than or equal to four.
 *
 * Lengths of one (4^0) are handled by the radix-2 path, which is a plain copy.
 */
bool is_power_of_four(std::size_t n) {
    return n >= 4 && std::has_single_bit(n) && (std::countr_zero(n) % 2 == 0);
}

/**
 * @brief Selects the algorithm for a transform length.
 *
 * Radix4 for powers of four, Radix2 for other powers of two, DirectDFT otherwise.
 */
Algorithm select_algorithm(std::size_t n) {
    if (is_power_of_four(n))
        return Algorithm::Radix4;
    if (std::has_single_bit(n))
        return Algorithm::Radix2;
    return Algorithm::DirectDFT;
}

} // namespace


/**
 * @brief Private state of an FFTPlan. Null in the owning object once moved from.
 *
 * Holds the precomputed twiddle factors exp(-2*pi*i*k/N) for k in [0, N),
 * computed in double precision regardless of T.
 */
template<typename T>
struct FFTPlan<T>::Impl {
    /**
     * @brief Validates the length and precomputes the twiddle table.
     * @throws InvalidSizeException if @p n is zero.
     */
    Impl(std::size_t n, Normalization norm, Algorithm alg)
        : length(n), normalization(norm), algorithm(alg) {
        if (length == 0)
            throw InvalidSizeException("FFT length must be greater than zero");

        twiddles.reserve(length);
        const double step = 2.0 * std::numbers::pi / static_cast<double>(length);
        for (std::size_t k = 0; k < length; ++k)
            twiddles.push_back(std::polar(1.0, -step * static_cast<double>(k)));
    }

    std::size_t length;
    Normalization normalization;
    Algorithm algorithm;
    std::vector<std::complex<double>> twiddles;
};

template<typename T>
FFTPlan<T>::FFTPlan(std::size_t size, Normalization normalization)
    : impl_(std::make_unique<Impl>(size, normalization, select_algorithm(size))) {}

template<typename T>
FFTPlan<T>::~FFTPlan() = default;

template<typename T>
FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;

template<typename T>
FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan&& other) noexcept = default;

template<typename T>
std::size_t FFTPlan<T>::size() const noexcept {
    return impl_ ? impl_->length : 0;
}

template<typename T>
Shape<1> FFTPlan<T>::shape() const {
    if (!impl_)
        throw FFTException("FFTPlan::shape() called on a moved-from plan");
    return Shape<1>({impl_->length});
}

template<typename T>
Normalization FFTPlan<T>::normalization() const noexcept {
    return impl_ ? impl_->normalization : Normalization::None;
}

template<typename T>
Precision FFTPlan<T>::precision() const noexcept {
    if constexpr (std::is_same_v<T, float>)
        return Precision::Float32;
    else
        return Precision::Float64;
}

template<typename T>
Algorithm FFTPlan<T>::algorithm() const noexcept {
    return impl_ ? impl_->algorithm : Algorithm::Unselected;
}

namespace {

/**
 * @brief Returns the scale factor applied to the result of a transform.
 * @param normalization Plan normalization convention.
 * @param direction     Transform direction.
 * @param n            Transform length.
 */
double scale_factor(Normalization normalization, Direction direction, std::size_t n) {
    const double dn = static_cast<double>(n);
    const bool inverse = direction == Direction::Inverse;
    switch (normalization) {
        case Normalization::None:    return 1.0;
        case Normalization::Forward: return inverse ? 1.0 : 1.0 / dn;
        case Normalization::Inverse: return inverse ? 1.0 / dn : 1.0;
        case Normalization::Ortho:   return 1.0 / std::sqrt(dn);
    }
    return 1.0;
}

/**
 * @brief Returns true if the byte ranges [a, a+bytes) and [b, b+bytes) intersect.
 */
bool ranges_overlap(const void* a, const void* b, std::size_t bytes) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + bytes && pb < pa + bytes;
}

/**
 * @brief Direct O(N^2) DFT, computed in double precision.
 * @param twiddles Table of exp(-2*pi*i*k/N), k in [0, N).
 * @param inverse  True for the inverse transform (conjugated twiddles).
 * @param in       Input samples, N elements, not aliased with @p out.
 * @param out      Output spectrum, N elements, before scaling.
 */
template<typename T>
void direct_dft(const std::vector<std::complex<double>>& twiddles, bool inverse,
                const std::complex<T>* in, std::complex<T>* out) {
    const std::size_t n = twiddles.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const std::complex<double> w = twiddles[(k * j) % n];
            acc += std::complex<double>(in[j]) * (inverse ? std::conj(w) : w);
        }
        out[k] = std::complex<T>(static_cast<T>(acc.real()), static_cast<T>(acc.imag()));
    }
}

/**
 * @brief In-place iterative radix-2 decimation-in-time FFT (Cooley-Tukey).
 * @param twiddles Table of exp(-2*pi*i*k/N), k in [0, N); N must be a power of two.
 * @param inverse  True for the inverse transform (conjugated twiddles).
 * @param data     N elements, transformed in place, before scaling.
 */
template<typename T>
void radix2_in_place(const std::vector<std::complex<double>>& twiddles, bool inverse,
                     std::complex<T>* data) {
    const std::size_t n = twiddles.size();

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }

    // Butterfly stages: combine pairs of sub-transforms of length len/2.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const std::size_t stride = n / len;
        for (std::size_t start = 0; start < n; start += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const std::complex<double> wd = twiddles[k * stride];
                const std::complex<T> w(static_cast<T>(wd.real()),
                                        static_cast<T>(inverse ? -wd.imag() : wd.imag()));
                const std::complex<T> u = data[start + k];
                const std::complex<T> v = data[start + k + half] * w;
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
    }
}


/**
 * @brief Permutes @p data into base-4 digit-reversed order, in place.
 * @param data N elements, N a power of four.
 */
template<typename T>
void digit_reverse_base4(std::complex<T>* data, std::size_t n) {
    const std::size_t digits = static_cast<std::size_t>(std::countr_zero(n)) / 2;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t r = 0;
        std::size_t x = i;
        for (std::size_t d = 0; d < digits; ++d) {
            r = (r << 2) | (x & 3u);
            x >>= 2;
        }
        if (i < r)
            std::swap(data[i], data[r]);
    }
}

/**
 * @brief In-place iterative radix-4 decimation-in-time FFT.
 *
 * Each stage combines four sub-transforms of length q = len/4 using the
 * identity X[k + m*q] = sum_r W_len^(r*k) * Y_r[k] * (W_4^(r*m)), where
 * W_4 = -i for the forward transform and +i for the inverse.
 *
 * @param twiddles Table of exp(-2*pi*i*k/N), k in [0, N); N must be a power of four.
 * @param inverse  True for the inverse transform (conjugated twiddles).
 * @param data     N elements, transformed in place, before scaling.
 */
template<typename T>
void radix4_in_place(const std::vector<std::complex<double>>& twiddles, bool inverse,
                     std::complex<T>* data) {
    const std::size_t n = twiddles.size();
    digit_reverse_base4(data, n);

    // Multiplication by -i (forward) or +i (inverse) is a swap and sign change,
    // so it is done without a full complex product.
    auto rotate = [inverse](std::complex<T> d) {
        return inverse ? std::complex<T>(-d.imag(), d.real())
                       : std::complex<T>(d.imag(), -d.real());
    };

    for (std::size_t len = 4; len <= n; len <<= 2) {
        const std::size_t q = len >> 2;
        const std::size_t stride = n / len;
        for (std::size_t k = 0; k < q; ++k) {
            // Twiddles depend only on k and the stage, so compute them once per k.
            // The largest index 3*k*stride is below n, so no modulo is needed.
            auto tw = [&](std::size_t power) {
                const std::complex<double> wd = twiddles[power * k * stride];
                return std::complex<T>(static_cast<T>(wd.real()),
                                       static_cast<T>(inverse ? -wd.imag() : wd.imag()));
            };
            const std::complex<T> w1 = tw(1);
            const std::complex<T> w2 = tw(2);
            const std::complex<T> w3 = tw(3);

            for (std::size_t start = 0; start < n; start += len) {
                const std::complex<T> a0 = data[start + k];
                const std::complex<T> a1 = data[start + k + q] * w1;
                const std::complex<T> a2 = data[start + k + 2 * q] * w2;
                const std::complex<T> a3 = data[start + k + 3 * q] * w3;

                const std::complex<T> t0 = a0 + a2;
                const std::complex<T> t1 = a0 - a2;
                const std::complex<T> t2 = a1 + a3;
                const std::complex<T> t3 = rotate(a1 - a3);

                data[start + k]         = t0 + t2;
                data[start + k + q]     = t1 + t3;
                data[start + k + 2 * q] = t0 - t2;
                data[start + k + 3 * q] = t1 - t3;
            }
        }
    }
}

} // namespace

template<typename T>
void FFTPlan<T>::execute(Direction direction,
                         const std::complex<T>* in,
                         std::complex<T>* out) const {
    if (!impl_)
        throw FFTException("FFTPlan::execute() called on a moved-from plan");
    if (in == nullptr || out == nullptr)
        throw InvalidBufferException("FFT input and output buffers must not be null");

    const std::size_t n = impl_->length;
    if (ranges_overlap(in, out, n * sizeof(std::complex<T>)))
        throw InvalidBufferException("FFT input and output buffers must not overlap");

    const bool inverse = direction == Direction::Inverse;

    switch (impl_->algorithm) {
        case Algorithm::Radix4:
            std::copy(in, in + n, out);
            radix4_in_place(impl_->twiddles, inverse, out);
            break;
        case Algorithm::Radix2:
            std::copy(in, in + n, out);
            radix2_in_place(impl_->twiddles, inverse, out);
            break;
        default:
            direct_dft(impl_->twiddles, inverse, in, out);
            break;
    }

    const double scale = scale_factor(impl_->normalization, direction, n);
    if (scale != 1.0) {
        const auto s = static_cast<T>(scale);
        for (std::size_t k = 0; k < n; ++k)
            out[k] *= s;
    }
}

// Explicit instantiation definitions. Export comes from the class definition.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace fft
