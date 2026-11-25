#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include "fft_export.hpp"
#include "shape.hpp"
#include "types.hpp"

namespace fft {

template<typename T>
class FFTPlan {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "FFTPlan supports float and double only");

public:
    explicit FFTPlan(std::size_t size,
                     Normalization normalization = Normalization::Inverse);

    ~FFTPlan();

    FFTPlan(FFTPlan&&) noexcept;
    FFTPlan& operator=(FFTPlan&&) noexcept;

    FFTPlan(const FFTPlan&) = delete;
    FFTPlan& operator=(const FFTPlan&) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Shape<1> shape() const;
    [[nodiscard]] Normalization normalization() const noexcept;
    [[nodiscard]] Precision precision() const noexcept;
    [[nodiscard]] Algorithm algorithm() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class FFT_API FFTPlan<float>;
extern template class FFT_API FFTPlan<double>;

} // namespace fft
