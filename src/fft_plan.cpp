#include "fft/fft_plan.hpp"

#include <type_traits>
#include <utility>

#include "fft/exceptions.hpp"

namespace fft {

/**
 * @brief Private state of an FFTPlan. Null in the owning object once moved from.
 */
template<typename T>
struct FFTPlan<T>::Impl {
    /**
     * @brief Validates and stores the transform length.
     * @throws InvalidSizeException if @p n is zero.
     */
    explicit Impl(std::size_t n, Normalization norm)
        : length(n), normalization(norm) {
        if (length == 0)
            throw InvalidSizeException("FFT length must be greater than zero");
    }

    std::size_t length;
    Normalization normalization;
};

template<typename T>
FFTPlan<T>::FFTPlan(std::size_t size, Normalization normalization)
    : impl_(std::make_unique<Impl>(size, normalization)) {}

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
    return Algorithm::Unselected;
}

// Explicit instantiation definitions. Export comes from the class definition.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace fft
