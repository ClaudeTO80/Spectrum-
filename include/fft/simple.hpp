#pragma once

/**
 * @file simple.hpp
 * @brief Basic API: one-call forward and inverse transforms.
 *
 * These functions build a plan, run it and return a new vector. The real
 * transforms (forward_real, inverse_real) take a real sequence and return the
 * N/2+1 non-redundant spectrum bins. They use the
 * default normalization (forward unscaled, inverse divided by N), so that
 * Spectrum::inverse(Spectrum::forward(x)) reproduces x.
 *
 * Each call builds a new plan, so it recomputes the twiddle table. Lengths that
 * use the mixed-radix, Bluestein or Rader algorithms also allocate scratch
 * memory on each call. For repeated transforms of the same length, create an
 * FFTPlan once and reuse it, with a Workspace if allocations matter (see
 * fft_plan.hpp).
 */

#include <complex>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include "fft_plan.hpp"
#include "nd_plan.hpp"
#include "real_plan.hpp"
#include "types.hpp"

namespace Spectrum {

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

/**
 * @brief Forward transform of a real sequence, returning its N/2+1 spectrum bins.
 * @tparam T    Precision, @c float or @c double.
 * @param input Real samples; must not be empty.
 * @return Bins X[0] .. X[N/2] of the unscaled-forward, 1/N-inverse convention.
 * @throws InvalidSizeException if @p input is empty.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> forward_real(std::span<const T> input) {
    RealFFTPlan<T> plan(input.size(), Normalization::Inverse);
    std::vector<std::complex<T>> output(plan.spectrum_size());
    plan.execute_forward(input.data(), output.data());
    return output;
}

/**
 * @brief Inverse transform back to a real sequence of length @p n, scaled by 1/n.
 * @tparam T     Precision, @c float or @c double.
 * @param spectrum The n/2+1 bins produced by forward_real for a length-@p n input.
 * @param n      Length of the real sequence to reconstruct; must be greater than zero.
 * @return The reconstructed real sequence of length @p n.
 * @throws InvalidSizeException if @p n is zero or @p spectrum does not hold n/2+1 bins.
 *
 * The imaginary parts of the DC and Nyquist bins are ignored.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<T> inverse_real(std::span<const std::complex<T>> spectrum,
                                          std::size_t n) {
    if (n == 0)
        throw InvalidSizeException("inverse_real: output length must be greater than zero");
    if (spectrum.size() != n / 2 + 1)
        throw InvalidSizeException("inverse_real: spectrum must hold n/2+1 bins");
    RealFFTPlan<T> plan(n, Normalization::Inverse);
    std::vector<T> output(n);
    plan.execute_inverse(spectrum.data(), output.data());
    return output;
}

/** @brief Overload for std::vector; see forward_real(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> forward_real(const std::vector<T>& input) {
    return forward_real(std::span<const T>(input));
}

/** @brief Overload for std::vector; see inverse_real(std::span, std::size_t). */
template<std::floating_point T>
[[nodiscard]] std::vector<T> inverse_real(const std::vector<std::complex<T>>& spectrum,
                                          std::size_t n) {
    return inverse_real(std::span<const std::complex<T>>(spectrum), n);
}

/**
 * @brief Forward 2D transform of a row-major rows x cols grid.
 * @throws InvalidSizeException if a dimension is zero or input.size() != rows*cols.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> forward_2d(std::span<const std::complex<T>> input,
                                                      std::size_t rows, std::size_t cols) {
    FFTPlanND<T, 2> plan(Shape<2>({rows, cols}), Normalization::Inverse);
    if (input.size() != plan.size())
        throw InvalidSizeException("forward_2d: input must hold rows*cols elements");
    std::vector<std::complex<T>> output(input.begin(), input.end());
    plan.execute_inplace(Direction::Forward, output.data());
    return output;
}

/**
 * @brief Inverse 2D transform, scaled by 1/(rows*cols).
 * @throws InvalidSizeException if a dimension is zero or input.size() != rows*cols.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> inverse_2d(std::span<const std::complex<T>> input,
                                                      std::size_t rows, std::size_t cols) {
    FFTPlanND<T, 2> plan(Shape<2>({rows, cols}), Normalization::Inverse);
    if (input.size() != plan.size())
        throw InvalidSizeException("inverse_2d: input must hold rows*cols elements");
    std::vector<std::complex<T>> output(input.begin(), input.end());
    plan.execute_inplace(Direction::Inverse, output.data());
    return output;
}

/**
 * @brief Forward 3D transform of a row-major rows x cols x depth grid.
 * @throws InvalidSizeException if a dimension is zero or input.size() != rows*cols*depth.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> forward_3d(std::span<const std::complex<T>> input,
                                                      std::size_t rows, std::size_t cols,
                                                      std::size_t depth) {
    FFTPlanND<T, 3> plan(Shape<3>({rows, cols, depth}), Normalization::Inverse);
    if (input.size() != plan.size())
        throw InvalidSizeException("forward_3d: input must hold rows*cols*depth elements");
    std::vector<std::complex<T>> output(input.begin(), input.end());
    plan.execute_inplace(Direction::Forward, output.data());
    return output;
}

/**
 * @brief Inverse 3D transform, scaled by 1/(rows*cols*depth).
 * @throws InvalidSizeException if a dimension is zero or input.size() != rows*cols*depth.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> inverse_3d(std::span<const std::complex<T>> input,
                                                      std::size_t rows, std::size_t cols,
                                                      std::size_t depth) {
    FFTPlanND<T, 3> plan(Shape<3>({rows, cols, depth}), Normalization::Inverse);
    if (input.size() != plan.size())
        throw InvalidSizeException("inverse_3d: input must hold rows*cols*depth elements");
    std::vector<std::complex<T>> output(input.begin(), input.end());
    plan.execute_inplace(Direction::Inverse, output.data());
    return output;
}

/** @brief Overload for std::vector; see forward_2d(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> forward_2d(const std::vector<std::complex<T>>& input,
                                                      std::size_t rows, std::size_t cols) {
    return forward_2d(std::span<const std::complex<T>>(input), rows, cols);
}

/** @brief Overload for std::vector; see inverse_2d(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> inverse_2d(const std::vector<std::complex<T>>& input,
                                                      std::size_t rows, std::size_t cols) {
    return inverse_2d(std::span<const std::complex<T>>(input), rows, cols);
}

/** @brief Overload for std::vector; see forward_3d(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> forward_3d(const std::vector<std::complex<T>>& input,
                                                      std::size_t rows, std::size_t cols,
                                                      std::size_t depth) {
    return forward_3d(std::span<const std::complex<T>>(input), rows, cols, depth);
}

/** @brief Overload for std::vector; see inverse_3d(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> inverse_3d(const std::vector<std::complex<T>>& input,
                                                      std::size_t rows, std::size_t cols,
                                                      std::size_t depth) {
    return inverse_3d(std::span<const std::complex<T>>(input), rows, cols, depth);
}

} // namespace Spectrum
