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
    : impl_(std::make_unique<Impl>(
          size, normalization,
          std::has_single_bit(size) ? Algorithm::Radix2 : Algorithm::DirectDFT)) {}

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

    if (impl_->algorithm == Algorithm::Radix2) {
        std::copy(in, in + n, out);
        radix2_in_place(impl_->twiddles, inverse, out);
    } else {
        direct_dft(impl_->twiddles, inverse, in, out);
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
