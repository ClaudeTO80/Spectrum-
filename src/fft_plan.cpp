#include "fft/fft_plan.hpp"

#include <type_traits>
#include <utility>

#include "fft/exceptions.hpp"

namespace fft {

template<typename T>
struct FFTPlan<T>::Impl {
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
FFTPlan<T>::FFTPlan(FFTPlan&&) noexcept = default;

template<typename T>
FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan&&) noexcept = default;

template<typename T>
std::size_t FFTPlan<T>::size() const noexcept {
    return impl_->length;
}

template<typename T>
Shape<1> FFTPlan<T>::shape() const {
    return Shape<1>({impl_->length});
}

template<typename T>
Normalization FFTPlan<T>::normalization() const noexcept {
    return impl_->normalization;
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

template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace fft
