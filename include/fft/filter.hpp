#pragma once

/**
 * @file filter.hpp
 * @brief Frequency-domain filtering of real signals and real 2D planes
 *        (for example grayscale images): forward transform, spectral mask,
 *        inverse transform.
 *
 * The filters keep or remove whole spectrum bins.
 *
 * 1D: bin k, for k = 0 .. N/2, of an N-sample record sampled at rate fs is
 * associated with frequency f_k = k * fs / N (see filter_real).
 *
 * 2D: bin (ky, kx) of a rows x cols plane is associated with the radial
 * frequency r = sqrt(fx^2 + fy^2), where fx = min(kx, cols-kx) / cols and
 * fy = min(ky, rows-ky) / rows, measured in cycles per pixel
 * (see filter_real_2d).
 *
 * Two limitations follow from the mask being brick-wall (ideal):
 *  - a sinusoid whose frequency is not an integer multiple of the bin spacing
 *    spreads over neighbouring bins (spectral leakage), so part of its energy
 *    can be removed or kept on the wrong side of a cutoff;
 *  - a sharp cutoff produces ringing in the time or space domain (Gibbs
 *    phenomenon), visible as halos along edges in images.
 * A sinusoid with an integer number of cycles over the record is separated
 * exactly.
 */

#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <span>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "real_plan.hpp"
#include "shape.hpp"
#include "simple.hpp"
#include "types.hpp"

namespace Spectrum {

/**
 * @brief Shape of the spectral mask applied by the filter functions.
 *
 * Boundaries are inclusive on the kept side, so a low-pass and a high-pass
 * with the same cutoff are exact complements of each other, and so are a
 * band-pass and a band-stop with the same limits.
 */
enum class FilterType : std::uint8_t {
    LowPass,  ///< Keeps f <= cutoff_low.
    HighPass, ///< Keeps f > cutoff_low.
    BandPass, ///< Keeps cutoff_low <= f <= cutoff_high.
    BandStop  ///< Keeps f < cutoff_low or f > cutoff_high.
};

namespace detail {

/**
 * @brief Whether a frequency @p f survives the mask of the given type.
 */
template<std::floating_point T>
[[nodiscard]] constexpr bool keep_frequency(FilterType type, T f, T lo, T hi) noexcept {
    switch (type) {
    case FilterType::LowPass:
        return f <= lo;
    case FilterType::HighPass:
        return f > lo;
    case FilterType::BandPass:
        return f >= lo && f <= hi;
    case FilterType::BandStop:
        return f < lo || f > hi;
    }
    return false;
}

/**
 * @brief Validates cutoffs against the admissible range [0, max_cutoff].
 *
 * Written as negated comparisons so that NaN is rejected too.
 * @param who        Name of the public function, used in the error message.
 * @param max_cutoff Largest admissible cutoff, in the same unit as the cutoffs.
 * @param limit_text Description of @p max_cutoff for the error message, for example "fs/2".
 * @throws InvalidSizeException if a cutoff is out of range or the band is inverted.
 */
template<std::floating_point T>
void validate_mask(const char* who, FilterType type, T max_cutoff,
                   const char* limit_text, T cutoff_low, T cutoff_high) {
    const std::string range = std::string("[0, ") + limit_text + "]";
    if (!(cutoff_low >= T{0} && cutoff_low <= max_cutoff))
        throw InvalidSizeException(std::string(who) + ": cutoff_low must lie in " + range);

    const bool is_band = (type == FilterType::BandPass || type == FilterType::BandStop);
    if (is_band && !(cutoff_high >= cutoff_low && cutoff_high <= max_cutoff))
        throw InvalidSizeException(std::string(who) + ": cutoff_high must lie in [cutoff_low, " +
                                   limit_text + "]");
}

} // namespace detail

/**
 * @brief Filters a real signal in the frequency domain.
 *
 * Applies the pipeline forward transform -> spectral mask -> inverse transform
 * using a RealFFTPlan of the input length, with the default normalization, so
 * a mask that keeps every bin returns the input unchanged.
 *
 * @tparam T          Precision, @c float or @c double.
 * @param input       Real samples; must not be empty.
 * @param sample_rate Sampling frequency fs, in the same unit as the cutoffs
 *                    (for example Hz); must be positive.
 * @param type        Shape of the mask.
 * @param cutoff_low  Lower cutoff for LowPass, HighPass, BandPass and BandStop.
 *                    Must lie in [0, fs/2].
 * @param cutoff_high Upper cutoff, used by BandPass and BandStop only. Must lie
 *                    in [cutoff_low, fs/2]. Ignored by the other types.
 * @return The filtered signal, of the same length as @p input.
 * @throws InvalidSizeException if @p input is empty, @p sample_rate is not
 *         positive, or a cutoff is outside its admissible range. NaN values
 *         are rejected.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<T> filter_real(std::span<const T> input,
                                         T sample_rate,
                                         FilterType type,
                                         T cutoff_low,
                                         T cutoff_high = T{0}) {
    if (input.empty())
        throw InvalidSizeException("filter_real: input must not be empty");
    if (!(sample_rate > T{0}))
        throw InvalidSizeException("filter_real: sample rate must be positive");

    const T nyquist = sample_rate / T{2};
    detail::validate_mask("filter_real", type, nyquist, "fs/2", cutoff_low, cutoff_high);

    const std::size_t n = input.size();
    RealFFTPlan<T> plan(n, Normalization::Inverse);

    std::vector<std::complex<T>> spectrum(plan.spectrum_size());
    plan.execute_forward(input.data(), spectrum.data());

    const T bin_width = sample_rate / static_cast<T>(n);
    for (std::size_t k = 0; k < spectrum.size(); ++k) {
        const T f = static_cast<T>(k) * bin_width;
        if (!detail::keep_frequency(type, f, cutoff_low, cutoff_high))
            spectrum[k] = std::complex<T>{T{0}, T{0}};
    }

    std::vector<T> output(n);
    plan.execute_inverse(spectrum.data(), output.data());
    return output;
}

/** @brief Overload for std::vector; see filter_real(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<T> filter_real(const std::vector<T>& input,
                                         T sample_rate,
                                         FilterType type,
                                         T cutoff_low,
                                         T cutoff_high = T{0}) {
    return filter_real(std::span<const T>(input), sample_rate, type, cutoff_low, cutoff_high);
}

/**
 * @brief Filters a real 2D plane (row-major, rows x cols) with a radial mask.
 *
 * Applies forward 2D transform -> radial spectral mask -> inverse 2D transform.
 * Frequencies are radii in cycles per pixel. The Nyquist frequency is 0.5 along
 * each axis, but the radius of a bin is sqrt(fx^2 + fy^2), which reaches
 * sqrt(0.5) at the corners of the spectrum of an even-sized plane. The admissible
 * cutoffs therefore lie in [0, sqrt(0.5)], and only a cutoff of sqrt(0.5) keeps
 * all bins.
 *
 * A mask that keeps every bin returns the input unchanged, up to rounding.
 *
 * @tparam T    Precision, @c float or @c double.
 * @param input Samples, row-major, of length rows*cols.
 * @param rows  Number of rows; must be greater than zero.
 * @param cols  Number of columns; must be greater than zero.
 * @param type  Shape of the mask.
 * @param cutoff_low  Lower cutoff in cycles per pixel, in [0, sqrt(0.5)].
 * @param cutoff_high Upper cutoff for BandPass and BandStop, in [cutoff_low, sqrt(0.5)].
 *                    Ignored by the other types.
 * @return The filtered plane, row-major, of the same size as @p input.
 * @throws InvalidSizeException if a dimension is zero, @p input does not hold
 *         rows*cols samples, or a cutoff is outside its admissible range.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<T> filter_real_2d(std::span<const T> input,
                                            std::size_t rows,
                                            std::size_t cols,
                                            FilterType type,
                                            T cutoff_low,
                                            T cutoff_high = T{0}) {
    const Shape<2> shape({rows, cols}); // throws if a dimension is zero
    if (input.size() != shape.element_count())
        throw InvalidSizeException("filter_real_2d: input must hold rows*cols samples");

    // Largest radius of a bin: the corner (0.5, 0.5) of an even-sized plane.
    detail::validate_mask("filter_real_2d", type, std::sqrt(T{0.5}), "sqrt(0.5)",
                          cutoff_low, cutoff_high);

    std::vector<std::complex<T>> plane(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        plane[i] = std::complex<T>{input[i], T{0}};

    auto spectrum = forward_2d(std::span<const std::complex<T>>(plane), rows, cols);

    const T inv_rows = T{1} / static_cast<T>(rows);
    const T inv_cols = T{1} / static_cast<T>(cols);
    for (std::size_t ky = 0; ky < rows; ++ky) {
        const T fy = static_cast<T>(std::min(ky, rows - ky)) * inv_rows;
        for (std::size_t kx = 0; kx < cols; ++kx) {
            const T fx = static_cast<T>(std::min(kx, cols - kx)) * inv_cols;
            const T radius = std::sqrt(fx * fx + fy * fy);
            if (!detail::keep_frequency(type, radius, cutoff_low, cutoff_high))
                spectrum[ky * cols + kx] = std::complex<T>{T{0}, T{0}};
        }
    }

    auto back = inverse_2d(std::span<const std::complex<T>>(spectrum), rows, cols);
    std::vector<T> output(input.size());
    for (std::size_t i = 0; i < output.size(); ++i)
        output[i] = back[i].real();
    return output;
}

/** @brief Overload for std::vector; see filter_real_2d(std::span). */
template<std::floating_point T>
[[nodiscard]] std::vector<T> filter_real_2d(const std::vector<T>& input,
                                            std::size_t rows,
                                            std::size_t cols,
                                            FilterType type,
                                            T cutoff_low,
                                            T cutoff_high = T{0}) {
    return filter_real_2d(std::span<const T>(input), rows, cols, type, cutoff_low, cutoff_high);
}

} // namespace Spectrum
