/**
 * @file test_filter_stress.cpp
 * @brief Stress tests of the frequency-domain filters on dense, large-N signals.
 *
 * Each case builds a clean signal and the same signal with additive white noise,
 * then runs the filter on both and checks two things:
 *
 *  - Quality, on the clean signal. Its tones lie on exact bins, so the ideal
 *    low-pass output is known analytically. The maximum error of the transform
 *    round trip is bounded.
 *
 *  - Quality, on the noisy signal. The filter is linear and the reference is
 *    exact, so the filtered error equals the filtered noise. For white noise of
 *    standard deviation sigma, a brick-wall low-pass keeps a fraction of the
 *    bins, and the RMS of the filtered noise must be sigma * sqrt(kept fraction).
 *    The measured ratio is compared with that prediction.
 *
 * Timings are printed for the filter and for a bare forward/inverse round trip
 * with a reused plan, so that the cost of the mask itself can be read off. The
 * timing guard is a loose ceiling that only catches gross regressions: absolute
 * numbers depend on the machine, and only a Release build gives representative
 * performance.
 */

#include <fft/fft.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int g_failures = 0;

/// Records a failed check with its location.
#define FILTER_STRESS_CHECK(cond)                                                      \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": required: " #cond "\n";     \
            ++g_failures;                                                              \
        }                                                                              \
    } while (false)

/// White noise uniform on [-a, a) has standard deviation a / sqrt(3).
constexpr double kNoiseHalfWidth = 0.5;
constexpr double kNoiseSigma = kNoiseHalfWidth / std::numbers::sqrt3;

/// Loose ceiling for one filter call, in nanoseconds per sample. Catches gross
/// regressions; not a performance target. Release measured about 640 ns/sample
/// on the worst case (prime length, Bluestein), so the Release ceiling leaves a
/// factor of about 3. Debug runs several times slower, so its ceiling is 10x.
#ifdef NDEBUG
constexpr double kMaxNsPerSample = 2000.0;
#else
constexpr double kMaxNsPerSample = 20000.0;
#endif

/// Relative tolerance between the measured and predicted noise attenuation.
constexpr double kAttenuationTolerance = 0.02;

/// Tolerance on the maximum error of the clean-signal round trip.
constexpr double kCleanErrorTolerance = 1e-9;

/// Accumulates results so the optimizer cannot discard the timed calls.
volatile double g_sink = 0.0;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Runs @p f @p reps times and returns the fastest run, in milliseconds.
template<typename F>
double best_of(int reps, F&& f) {
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        const auto start = Clock::now();
        g_sink = g_sink + f();
        best = std::min(best, elapsed_ms(start));
    }
    return best;
}

/// Deterministic uniform noise on [-kNoiseHalfWidth, kNoiseHalfWidth).
std::vector<double> uniform_noise(std::size_t n, std::uint32_t seed) {
    std::mt19937 gen(seed); // fully specified by the standard, so reproducible
    std::vector<double> x(n);
    for (auto& v : x) {
        const double u = (static_cast<double>(gen()) + 0.5) / 4294967296.0; // (0, 1)
        v = kNoiseHalfWidth * (2.0 * u - 1.0);
    }
    return x;
}

double rms(std::span<const double> x) {
    long double acc = 0.0L;
    for (const double v : x)
        acc += static_cast<long double>(v) * v;
    return std::sqrt(static_cast<double>(acc / static_cast<long double>(x.size())));
}

double max_abs_diff(std::span<const double> a, std::span<const double> b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

std::vector<double> difference(std::span<const double> a, std::span<const double> b) {
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        d[i] = a[i] - b[i];
    return d;
}

/**
 * @brief Fraction of the n bins of a real transform kept by a low-pass.
 *
 * Mirrors the mask of filter_real: bin k has frequency min(k, n-k) * fs/n, and is
 * kept when that frequency is at most the cutoff.
 */
double kept_fraction_1d(std::size_t n, double fs, double cutoff) {
    const double bin_width = fs / static_cast<double>(n);
    std::size_t kept = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t m = std::min(k, n - k);
        if (static_cast<double>(m) * bin_width <= cutoff)
            ++kept;
    }
    return static_cast<double>(kept) / static_cast<double>(n);
}

/**
 * @brief Fraction of the rows*cols bins of a 2D transform kept by a radial low-pass.
 *
 * Mirrors the mask of filter_real_2d: the radius of bin (ky, kx) is measured in
 * cycles per pixel, with the aliased index min(k, N-k).
 */
double kept_fraction_2d(std::size_t rows, std::size_t cols, double cutoff) {
    const double inv_rows = 1.0 / static_cast<double>(rows);
    const double inv_cols = 1.0 / static_cast<double>(cols);
    std::size_t kept = 0;
    for (std::size_t ky = 0; ky < rows; ++ky) {
        const double fy = static_cast<double>(std::min(ky, rows - ky)) * inv_rows;
        for (std::size_t kx = 0; kx < cols; ++kx) {
            const double fx = static_cast<double>(std::min(kx, cols - kx)) * inv_cols;
            if (std::sqrt(fx * fx + fy * fy) <= cutoff)
                ++kept;
        }
    }
    return static_cast<double>(kept) / static_cast<double>(rows * cols);
}

/**
 * @brief 1D low-pass on n dense samples: quality and timing.
 *
 * Sampling is chosen so that bin k is k Hz (fs = n). The clean signal has tones at
 * 1000 Hz and 4000 Hz, which the low-pass keeps, and a tone at n/4 Hz, which it
 * removes. The cutoff is n/10 Hz.
 */
void check_1d(std::size_t n, const char* label, std::uint32_t seed) {
    using FT = Spectrum::FilterType;
    const double nd = static_cast<double>(n);
    const double fs = nd;
    const double cutoff = fs / 10.0;
    const double two_pi = 2.0 * std::numbers::pi;

    std::vector<double> reference(n), clean(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / nd;
        const double kept = std::cos(two_pi * 1000.0 * t) + 0.5 * std::sin(two_pi * 4000.0 * t);
        const double removed = 2.0 * std::cos(two_pi * static_cast<double>(n / 4) * t);
        reference[i] = kept;
        clean[i] = kept + removed;
    }

    const auto noise = uniform_noise(n, seed);
    std::vector<double> noisy(n);
    for (std::size_t i = 0; i < n; ++i)
        noisy[i] = clean[i] + noise[i];

    // Quality on the clean signal: the ideal output is the reference, exactly.
    const auto out_clean = Spectrum::filter_real(clean, fs, FT::LowPass, cutoff);
    const double err_clean = max_abs_diff(out_clean, reference);
    FILTER_STRESS_CHECK(err_clean < kCleanErrorTolerance);

    // Quality on the noisy signal: the filtered error is the filtered noise.
    const auto out_noisy = Spectrum::filter_real(noisy, fs, FT::LowPass, cutoff);
    const double error_in = rms(difference(noisy, reference));
    const double error_out = rms(difference(out_noisy, reference));

    const double kept = kept_fraction_1d(n, fs, cutoff);
    const double predicted_ratio = std::sqrt(kept);
    const double measured_ratio = error_out / kNoiseSigma;
    const double deviation = std::abs(measured_ratio / predicted_ratio - 1.0);
    FILTER_STRESS_CHECK(deviation < kAttenuationTolerance);

    // Timing: the filter as a whole, and a bare round trip with a reused plan.
    const double ms_filter = best_of(3, [&] {
        const auto out = Spectrum::filter_real(noisy, fs, FT::LowPass, cutoff);
        return out[n / 2];
    });
    const double ns_per_sample = ms_filter * 1e6 / nd;
    FILTER_STRESS_CHECK(ns_per_sample < kMaxNsPerSample);

    Spectrum::RealFFTPlan<double> plan(n, Spectrum::Normalization::Inverse);
    std::vector<std::complex<double>> spectrum(plan.spectrum_size());
    std::vector<double> back(n);
    const double ms_roundtrip = best_of(3, [&] {
        plan.execute_forward(noisy.data(), spectrum.data());
        plan.execute_inverse(spectrum.data(), back.data());
        return back[n / 2];
    });

    std::printf("%-34s err(clean) %.2e | noise ratio %.4f vs %.4f (dev %.2f%%) | "
                "rms err %.4f -> %.4f | filter %.1f ms (%.0f ns/sample), round trip %.1f ms\n",
                label, err_clean, measured_ratio, predicted_ratio, 100.0 * deviation,
                error_in, error_out, ms_filter, ns_per_sample, ms_roundtrip);
}

/**
 * @brief 2D radial low-pass on a dense rows x cols plane: quality and timing.
 *
 * The clean plane is the sum of a cosine along the columns, 37 cycles across the
 * width (radius 37/cols, kept by the cutoff 0.05), and a cosine along the rows,
 * 700 cycles down the height (radius about 0.34, removed).
 */
void check_2d(std::size_t rows, std::size_t cols, const char* label, std::uint32_t seed) {
    using FT = Spectrum::FilterType;
    const double cutoff = 0.05; // cycles per pixel
    const double two_pi = 2.0 * std::numbers::pi;
    const std::size_t pixels = rows * cols;

    std::vector<double> reference(pixels), clean(pixels);
    for (std::size_t y = 0; y < rows; ++y) {
        for (std::size_t x = 0; x < cols; ++x) {
            const double kept = std::cos(two_pi * 37.0 * static_cast<double>(x) / static_cast<double>(cols));
            const double removed = std::cos(two_pi * 700.0 * static_cast<double>(y) / static_cast<double>(rows));
            reference[y * cols + x] = kept;
            clean[y * cols + x] = kept + removed;
        }
    }

    const auto noise = uniform_noise(pixels, seed);
    std::vector<double> noisy(pixels);
    for (std::size_t i = 0; i < pixels; ++i)
        noisy[i] = clean[i] + noise[i];

    const auto out_clean = Spectrum::filter_real_2d(clean, rows, cols, FT::LowPass, cutoff);
    const double err_clean = max_abs_diff(out_clean, reference);
    FILTER_STRESS_CHECK(err_clean < kCleanErrorTolerance);

    const auto out_noisy = Spectrum::filter_real_2d(noisy, rows, cols, FT::LowPass, cutoff);
    const double error_in = rms(difference(noisy, reference));
    const double error_out = rms(difference(out_noisy, reference));

    const double kept = kept_fraction_2d(rows, cols, cutoff);
    const double predicted_ratio = std::sqrt(kept);
    const double measured_ratio = error_out / kNoiseSigma;
    const double deviation = std::abs(measured_ratio / predicted_ratio - 1.0);
    FILTER_STRESS_CHECK(deviation < kAttenuationTolerance);

    const double ms_filter = best_of(2, [&] {
        const auto out = Spectrum::filter_real_2d(noisy, rows, cols, FT::LowPass, cutoff);
        return out[pixels / 2];
    });
    const double ns_per_pixel = ms_filter * 1e6 / static_cast<double>(pixels);
    FILTER_STRESS_CHECK(ns_per_pixel < kMaxNsPerSample);

    std::printf("%-34s err(clean) %.2e | noise ratio %.4f vs %.4f (dev %.2f%%) | "
                "rms err %.4f -> %.4f | filter %.1f ms (%.0f ns/pixel)\n",
                label, err_clean, measured_ratio, predicted_ratio, 100.0 * deviation,
                error_in, error_out, ms_filter, ns_per_pixel);
}

} // namespace

int main() {
#ifdef NDEBUG
    std::printf("build: Release, timings are representative\n");
#else
    std::printf("build: Debug, timings are indicative only; use Release for performance\n");
#endif

    check_1d(std::size_t{1} << 20, "1d lowpass 2^20", 101u);
    check_1d(1000000, "1d lowpass 1e6 (mixed radix)", 102u);
    check_1d(999983, "1d lowpass 999983 (prime)", 103u);
    check_1d(std::size_t{1} << 22, "1d lowpass 2^22", 104u);
    check_2d(2048, 1536, "2d lowpass 2048x1536", 201u);

    if (g_failures != 0) {
        std::cerr << g_failures << " filter stress check(s) failed\n";
        return 1;
    }
    std::printf("all filter stress checks passed\n");
    return 0;
}
