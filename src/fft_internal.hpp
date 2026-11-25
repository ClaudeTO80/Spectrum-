#pragma once

/**
 * @file fft_internal.hpp
 * @brief Helpers shared by the library's translation units. Not part of the public API.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "fft/types.hpp"

namespace fft::detail {

/**
 * @brief Scale factor applied to the unnormalized result of a transform.
 * @param normalization Normalization convention of the plan.
 * @param direction     Direction of the transform.
 * @param n             Transform length.
 */
inline double scale_factor(Normalization normalization, Direction direction, std::size_t n) {
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
 * @brief Returns true if the byte ranges [a, a+abytes) and [b, b+bbytes) intersect.
 *
 * Empty ranges never overlap anything.
 */
inline bool ranges_overlap(const void* a, std::size_t abytes,
                           const void* b, std::size_t bbytes) {
    if (abytes == 0 || bbytes == 0)
        return false;
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + bbytes && pb < pa + abytes;
}

} // namespace fft::detail
