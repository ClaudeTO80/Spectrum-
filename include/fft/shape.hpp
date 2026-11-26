#pragma once

/**
 * @file shape.hpp
 * @brief Dimension descriptor for transforms of any rank.
 */

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "exceptions.hpp"

namespace Spectrum {

/**
 * @brief Extents of a transform of rank Rank.
 *
 * Only the 1D case is executed by the current implementation. The type is
 * shared with the planned 2D and 3D transforms.
 *
 * @tparam Rank Number of dimensions; must be positive.
 */
template<std::size_t Rank>
class Shape {
    static_assert(Rank > 0, "Shape rank must be positive");

public:
    /** @brief Storage type for the extents. */
    using dimensions_type = std::array<std::size_t, Rank>;

    /**
     * @brief Constructs a shape from its extents.
     * @param dimensions Extent of each axis; every extent must be greater than zero.
     * @throws InvalidSizeException if any extent is zero.
     */
    constexpr explicit Shape(dimensions_type dimensions)
        : dimensions_(dimensions) {
        for (const auto extent : dimensions_) {
            if (extent == 0)
                throw InvalidSizeException("FFT dimensions must be greater than zero");
        }
    }

    /**
     * @brief Extent along one axis.
     * @param axis Axis index, in [0, Rank).
     * @throws std::out_of_range if @p axis is out of range.
     */
    [[nodiscard]] constexpr std::size_t extent(std::size_t axis) const {
        return dimensions_.at(axis);
    }

    /** @brief All extents. */
    [[nodiscard]] constexpr const dimensions_type& dimensions() const noexcept {
        return dimensions_;
    }

    /**
     * @brief Product of the extents: the number of elements of a transform of this shape.
     * @throws InvalidSizeException if the product overflows std::size_t.
     */
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

} // namespace Spectrum
