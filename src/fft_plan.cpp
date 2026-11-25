#include "fft/fft_plan.hpp"

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
    : impl_(std::make_unique<Impl>(size, normalization, Algorithm::DirectDFT)) {}

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

/**
 * @brief Returns the scale factor applied to the result of a transform.
 * @param normalization Plan normalization convention.
 * @param direction     Transform direction.
 * @param n            Transform length.
 */
namespace {

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
    const double scale = scale_factor(impl_->normalization, direction, n);
    const auto& twiddles = impl_->twiddles;

    // X[k] = scale * sum_j x[j] * W^(k*j), with W = exp(-2*pi*i/N) for Forward.
    // The index (k*j) mod N keeps the twiddle angle within one period.
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const std::complex<double> w = twiddles[(k * j) % n];
            acc += std::complex<double>(in[j]) * (inverse ? std::conj(w) : w);
        }
        out[k] = std::complex<T>(static_cast<T>(acc.real() * scale),
                                 static_cast<T>(acc.imag() * scale));
    }
}

// Explicit instantiation definitions. Export comes from the class definition.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace fft
