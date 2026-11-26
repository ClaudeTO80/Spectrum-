#pragma once

/**
 * @file filter.hpp
 * @brief Frequency-domain filtering of real signals: forward transform,
 *        spectral mask, inverse transform.
 *
 * The filter keeps or removes whole spectrum bins. Bin k of an N-sample
 * record, sampled at rate fs, is associated with frequency f_k = k * fs / N.
 *
 * Two limitations follow from the mask being brick-wall (ideal):
 *  - a sinusoid whose frequency is not an integer multiple of fs / N spreads
 *    over neighbouring bins (spectral leakage), so part of its energy can be
 *    removed or kept on the wrong side of a cutoff;
 *  - a sharp cutoff produces ringing in the time domain (Gibbs phenomenon).
 * A sinusoid with an integer number of cycles over the record is separated
 * exactly.
 */

#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "exceptions.hpp"
#include "real_plan.hpp"
#include "types.hpp"

namespace Spectrum {

/**
 * @brief Shape of the spectral mask applied by filter_real().
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

    // Written as negated comparisons so that NaN is rejected too.
    const T nyquist = sample_rate / T{2};
    if (!(cutoff_low >= T{0} && cutoff_low <= nyquist))
        throw InvalidSizeException("filter_real: cutoff_low must lie in [0, fs/2]");

    const bool is_band = (type == FilterType::BandPass || type == FilterType::BandStop);
    if (is_band && !(cutoff_high >= cutoff_low && cutoff_high <= nyquist))
        throw InvalidSizeException("filter_real: cutoff_high must lie in [cutoff_low, fs/2]");

    const std::size_t n = input.size();
    RealFFTPlan<T> plan(n, Normalization::Inverse);

    std::vector<std::complex<T>> spectrum(plan.spectrum_size());
    plan.execute_forward(input.data(), spectrum.data());

    const T bin_width = sample_rate / static_cast<T>(n);
    for (std::size_t k = 0; k < spectrum.size(); ++k) {
        const T f = static_cast<T>(k) * bin_width;
        bool keep = false;
        switch (type) {
        case FilterType::LowPass:
            keep = (f <= cutoff_low);
            break;
        case FilterType::HighPass:
            keep = (f > cutoff_low);
            break;
        case FilterType::BandPass:
            keep = (f >= cutoff_low && f <= cutoff_high);
            break;
        case FilterType::BandStop:
            keep = (f < cutoff_low || f > cutoff_high);
            break;
        }
        if (!keep)
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

} // namespace Spectrum
