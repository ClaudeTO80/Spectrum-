#pragma once

/**
 * @file types.hpp
 * @brief Enumerations shared by the public API.
 *
 * The numeric values are part of the library's binary interface; do not reorder them.
 */

#include <cstddef>
#include <cstdint>

namespace fft {

/**
 * @brief Direction of a transform.
 *
 * Forward uses exp(-2*pi*i*k*n/N) as kernel; Inverse uses exp(+2*pi*i*k*n/N).
 */
enum class Direction : std::uint8_t {
    Forward, ///< Forward transform.
    Inverse  ///< Inverse transform.
};

/**
 * @brief Scaling convention applied by a plan.
 *
 * Each value names the direction that is divided by N; Ortho divides both
 * directions by sqrt(N), which makes the transform unitary.
 */
enum class Normalization : std::uint8_t {
    None,    ///< No scaling in either direction.
    Forward, ///< Divide the forward transform by N.
    Inverse, ///< Divide the inverse transform by N (default).
    Ortho    ///< Divide both directions by sqrt(N).
};

/**
 * @brief Algorithm selected by a plan at construction.
 *
 * Selection depends only on the length. See FFTPlan for the rules.
 */
enum class Algorithm : std::uint8_t {
    Unselected,  ///< Plan was moved from; no algorithm.
    DirectDFT,   ///< O(N^2) direct sum, used for small primes.
    Radix2,      ///< Iterative Cooley-Tukey, powers of two.
    Radix4,      ///< Iterative Cooley-Tukey, powers of four.
    MixedRadix,  ///< Recursive Cooley-Tukey over the prime factors.
    Bluestein,   ///< Chirp-z convolution, large primes and composites with a large factor.
    Rader        ///< Cyclic convolution, primes p with p-1 a power of two.
};

/**
 * @brief Floating-point precision of a plan.
 */
enum class Precision : std::uint8_t {
    Float32, ///< Single precision, plans of type float.
    Float64  ///< Double precision, plans of type double.
};

} // namespace fft
