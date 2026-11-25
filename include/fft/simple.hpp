#pragma once

/**
 * @file simple.hpp
 * @brief Basic API: one-call forward and inverse transforms.
 *
 * These functions build a plan, run it and return a new vector. They use the
 * default normalization (forward unscaled, inverse divided by N), so that
 * fft::inverse(fft::forward(x)) reproduces x.
 *
 * Each call builds a new plan, so it recomputes the twiddle table. Lengths that
 * use the mixed-radix, Bluestein or Rader algorithms also allocate scratch
 * memory on each call. For repeated transforms of the same length, create an
 * FFTPlan once and reuse it, with a Workspace if allocations matter (see
 * fft_plan.hpp).
 */

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

#include "fft_plan.hpp"
#include "types.hpp"

namespace fft {

/**
 * @brief Forward transform of a complex sequence.
 * @tparam T    Precision, @c float or @c double.
 * @param input Samples; must not be empty.
 * @return The transform, of the same length as @p input.
 * @throws InvalidSizeException if @p input is empty.
 */
template<typename T>
[[nodiscard]] std::vector<std::complex<T>> forward(std::span<const std::complex<T>> input) {
    FFTPlan<T> plan(input.size(), Normalization::Inverse);
    std::vector<std::complex<T>> output(input.size());
    plan.execute(Direction::Forward, input.data(), output.data());
    return output;
}

/**
 * @brief Inverse transform of a complex sequence, scaled by 1/N.
 * @tparam T    Precision, @c float or @c double.
 * @param input Spectrum; must not be empty.
 * @return The inverse transform, of the same length as @p input.
 * @throws InvalidSizeException if @p input is empty.
 */
template<typename T>
[[nodiscard]] std::vector<std::complex<T>> inverse(std::span<const std::complex<T>> input) {
    FFTPlan<T> plan(input.size(), Normalization::Inverse);
    std::vector<std::complex<T>> output(input.size());
    plan.execute(Direction::Inverse, input.data(), output.data());
    return output;
}

/** @brief Overload for std::vector; see forward(std::span). */
template<typename T>
[[nodiscard]] std::vector<std::complex<T>> forward(const std::vector<std::complex<T>>& input) {
    return forward(std::span<const std::complex<T>>(input));
}

/** @brief Overload for std::vector; see inverse(std::span). */
template<typename T>
[[nodiscard]] std::vector<std::complex<T>> inverse(const std::vector<std::complex<T>>& input) {
    return inverse(std::span<const std::complex<T>>(input));
}

} // namespace fft
