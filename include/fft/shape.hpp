#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "exceptions.hpp"

namespace fft {

template<std::size_t Rank>
class Shape {
    static_assert(Rank > 0, "Shape rank must be positive");

public:
    using dimensions_type = std::array<std::size_t, Rank>;

    constexpr explicit Shape(dimensions_type dimensions)
        : dimensions_(dimensions) {
        for (const auto extent : dimensions_) {
            if (extent == 0)
                throw InvalidSizeException("FFT dimensions must be greater than zero");
        }
    }

    [[nodiscard]] constexpr std::size_t extent(std::size_t axis) const {
        return dimensions_.at(axis);
    }

    [[nodiscard]] constexpr const dimensions_type& dimensions() const noexcept {
        return dimensions_;
    }

    [[nodiscard]] std::size_t element_count() const {
        std::size_t result = 1;
        for (const auto extent : dimensions_) {
            if (result > std::numeric_limits<std::size_t>::max() / extent)
                throw InvalidSizeException("FFT shape element count overflows size_t");
            result *= extent;
        }
        return result;
    }

private:
    dimensions_type dimensions_;
};

} // namespace fft
