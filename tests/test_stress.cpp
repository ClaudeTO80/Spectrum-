/**
 * @file test_stress.cpp
 * @brief Large-input stress tests for 1D, real-input and multidimensional transforms.
 *
 * The inputs here are far too large for the O(N^2) reference DFT used in
 * test_core.cpp. Correctness is therefore checked with three independent
 * properties instead:
 *  - round trip: inverse(forward(x)) reproduces x;
 *  - Parseval's identity: the energy of the spectrum equals N times the
 *    energy of the signal;
 *  - sampled reference: a few bins are recomputed directly in O(N) each.
 * Real-valued inputs also check Hermitian symmetry of the multidimensional
 * spectrum. Each case prints its measured errors and timings.
 */

#include <fft/fft.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <span>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using cld = std::complex<long double>;

int g_failures = 0;

/**
 * @brief Records a failed bound check and reports its location and value.
 */
void report_failure(const char* expression, double value, double limit, const char* file, int line) {
    std::cerr << file << ':' << line << ": " << expression << " = " << value
              << " exceeds limit " << limit << '\n';
    ++g_failures;
}

/**
 * @brief Checks that @p value is at most @p limit. A NaN value fails the check.
 */
#define FFT_STRESS_LE(value, limit)                                              \
    do {                                                                         \
        const double v_ = static_cast<double>(value);                            \
        const double l_ = static_cast<double>(limit);                            \
        if (!(v_ <= l_)) report_failure(#value, v_, l_, __FILE__, __LINE__);     \
    } while (false)

/**
 * @brief Checks a boolean condition and counts a failure when it is false.
 */
#define FFT_STRESS_REQUIRE(cond)                                                         \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << __FILE__ << ':' << __LINE__ << ": required: " #cond "\n";       \
            ++g_failures;                                                                \
        }                                                                                \
    } while (false)

/**
 * @brief Upper bounds on the relative errors of one case.
 */
struct Tolerances {
    double roundtrip;  ///< max|y - x| / max|x| after forward then inverse
    double parseval;   ///< |E_X / (N * E_x) - 1|
    double bins;       ///< sampled-bin error, relative to sqrt(N) * RMS(x)
};

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/**
 * @brief Deterministic value in [-1, 1), from the same generator as test_core.cpp.
 */
double next_uniform(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<double>(state) / 2147483648.0 - 1.0;
}

/**
 * @brief Deterministic complex input with zero-mean components.
 */
template <class T>
std::vector<std::complex<T>> make_complex(std::size_t n, std::uint32_t seed) {
    std::vector<std::complex<T>> x(n);
    std::uint32_t state = seed;
    for (auto& v : x) {
        const double re = next_uniform(state);
        const double im = next_uniform(state);
        v = {static_cast<T>(re), static_cast<T>(im)};
    }
    return x;
}

/**
 * @brief Deterministic real input, stored as complex values with zero imaginary part.
 */
std::vector<std::complex<double>> make_real_as_complex(std::size_t n, std::uint32_t seed) {
    std::vector<std::complex<double>> x(n);
    std::uint32_t state = seed;
    for (auto& v : x) {
        v = {next_uniform(state), 0.0};
    }
    return x;
}

/**
 * @brief Energy sum |x|^2, accumulated in long double.
 */
template <class T>
long double energy(std::span<const std::complex<T>> x) {
    long double e = 0;
    for (const auto& v : x) {
        const long double re = v.real();
        const long double im = v.imag();
        e += re * re + im * im;
    }
    return e;
}

/**
 * @brief Largest modulus of the elements of @p x, accumulated in long double.
 */
template <class T>
long double peak(std::span<const std::complex<T>> x) {
    long double m = 0;
    for (const auto& v : x)
        m = std::max(m, std::abs(cld(v.real(), v.imag())));
    return m;
}

/**
 * @brief Row-major linear index of a multi-index.
 */
template <std::size_t R>
std::size_t linear_index(const std::array<std::size_t, R>& dims, const std::array<std::size_t, R>& k) {
    std::size_t lin = 0;
    for (std::size_t d = 0; d < R; ++d)
        lin = lin * dims[d] + k[d];
    return lin;
}

/**
 * @brief Forward DFT value at one multi-index, computed directly in O(N).
 *
 * The phase is reduced modulo one full turn per axis before the trigonometric
 * call, which keeps the long double argument small even for large N.
 */
template <class T, std::size_t R>
cld sampled_dft(std::span<const std::complex<T>> x, const std::array<std::size_t, R>& dims,
                const std::array<std::size_t, R>& k) {
    cld acc = 0;
    for (std::size_t lin = 0; lin < x.size(); ++lin) {
        std::size_t rest = lin;
        long double turns = 0;
        for (std::size_t d = R; d-- > 0;) {
            const std::size_t idx = rest % dims[d];
            rest /= dims[d];
            turns += static_cast<long double>((idx * k[d]) % dims[d]) /
                     static_cast<long double>(dims[d]);
        }
        const long double angle = -2.0L * std::numbers::pi_v<long double> * turns;
        acc += cld(x[lin].real(), x[lin].imag()) * cld(std::cos(angle), std::sin(angle));
    }
    return acc;
}

/**
 * @brief Bin indices sampled for the reference check: DC, a low bin, a
 *        one-third bin and the highest bin along every axis.
 */
template <std::size_t R>
std::vector<std::array<std::size_t, R>> sample_indices(const std::array<std::size_t, R>& dims) {
    std::array<std::size_t, R> low{}, third{}, last{};
    for (std::size_t d = 0; d < R; ++d) {
        low[d] = (d + 1) % dims[d];
        third[d] = dims[d] / 3;
        last[d] = dims[d] - 1;
    }
    return {std::array<std::size_t, R>{}, low, third, last};
}

/**
 * @brief Largest relative error between the computed bins and the sampled reference.
 */
template <class T, std::size_t R>
double sampled_bin_error(std::span<const std::complex<T>> x, std::span<const std::complex<T>> X,
                         const std::array<std::size_t, R>& dims) {
    const long double rms = std::sqrt(energy(x) / static_cast<long double>(x.size()));
    const long double scale = std::sqrt(static_cast<long double>(x.size())) * rms;
    double worst = 0;
    for (const auto& k : sample_indices<R>(dims)) {
        const cld ref = sampled_dft(x, dims, k);
        const auto& got = X[linear_index(dims, k)];
        worst = std::max(worst, static_cast<double>(std::abs(cld(got.real(), got.imag()) - ref) / scale));
    }
    return worst;
}

/**
 * @brief Largest deviation from Hermitian symmetry, X[k] = conj(X[-k]), on the sampled bins.
 */
template <class T, std::size_t R>
double hermitian_error(std::span<const std::complex<T>> X, const std::array<std::size_t, R>& dims) {
    const long double scale = std::sqrt(energy(X));
    double worst = 0;
    for (const auto& k : sample_indices<R>(dims)) {
        std::array<std::size_t, R> neg{};
        for (std::size_t d = 0; d < R; ++d)
            neg[d] = (dims[d] - k[d]) % dims[d];
        const auto& a = X[linear_index(dims, k)];
        const auto& b = X[linear_index(dims, neg)];
        const cld diff = cld(a.real(), a.imag()) - std::conj(cld(b.real(), b.imag()));
        worst = std::max(worst, static_cast<double>(std::abs(diff) / scale));
    }
    return worst;
}

/**
 * @brief Largest element-wise difference, relative to the peak of the reference.
 */
template <class T>
double relative_max_error(std::span<const std::complex<T>> y, std::span<const std::complex<T>> x) {
    long double worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const cld d = cld(y[i].real(), y[i].imag()) - cld(x[i].real(), x[i].imag());
        worst = std::max(worst, std::abs(d));
    }
    return static_cast<double>(worst / peak(x));
}

/**
 * @brief Relative mismatch between the spectral and the signal energy.
 */
template <class T>
double parseval_error(std::span<const std::complex<T>> x, std::span<const std::complex<T>> X) {
    const long double ex = energy(x);
    const long double eX = energy(X);
    return static_cast<double>(std::abs(eX / (static_cast<long double>(x.size()) * ex) - 1.0L));
}

/**
 * @brief Prints one line with the measured metrics of a case.
 */
void print_metrics(const char* label, std::size_t total, double fwd_ms, double inv_ms,
                   double roundtrip, double parseval, double bins) {
    std::printf("%-34s N=%-10zu fwd %9.2f ms  inv %9.2f ms  roundtrip %.2e  parseval %.2e  bins %.2e\n",
                label, total, fwd_ms, inv_ms, roundtrip, parseval, bins);
}

/**
 * @brief Runs a complex transform of any rank and checks it with the three properties.
 *
 * @param real_input When true, the input has zero imaginary part, and the
 *        spectrum is also checked for Hermitian symmetry.
 */
template <class T, std::size_t R>
void check_complex(const std::array<std::size_t, R>& dims, std::uint32_t seed, const Tolerances& tol,
                   const char* label, bool real_input = false) {
    std::size_t total = 1;
    for (const auto d : dims)
        total *= d;

    const Spectrum::FFTPlanND<T, R> plan{Spectrum::Shape<R>(dims)};
    const auto x = real_input ? [&] {
        const auto r = make_real_as_complex(total, seed);
        std::vector<std::complex<T>> c(total);
        for (std::size_t i = 0; i < total; ++i)
            c[i] = {static_cast<T>(r[i].real()), T{0}};
        return c;
    }() : make_complex<T>(total, seed);

    std::vector<std::complex<T>> X(total), y(total);
    auto t0 = Clock::now();
    plan.execute(Spectrum::Direction::Forward, x.data(), X.data());
    const double fwd_ms = elapsed_ms(t0);
    t0 = Clock::now();
    plan.execute(Spectrum::Direction::Inverse, X.data(), y.data());
    const double inv_ms = elapsed_ms(t0);

    const double rt = relative_max_error<T>(y, x);
    const double pv = parseval_error<T>(x, X);
    double bins = sampled_bin_error<T, R>(x, X, dims);
    if (real_input)
        bins = std::max(bins, hermitian_error<T, R>(X, dims));

    print_metrics(label, total, fwd_ms, inv_ms, rt, pv, bins);
    FFT_STRESS_LE(rt, tol.roundtrip);
    FFT_STRESS_LE(pv, tol.parseval);
    FFT_STRESS_LE(bins, tol.bins);
}

/**
 * @brief Runs a one-dimensional complex transform and checks the selected algorithm.
 */
template <class T>
void check_1d_complex(std::size_t n, Spectrum::Algorithm expected, std::uint32_t seed,
                      const Tolerances& tol, const char* label) {
    const Spectrum::FFTPlan<T> plan(n);
    FFT_STRESS_REQUIRE(plan.algorithm() == expected);

    const auto x = make_complex<T>(n, seed);
    std::vector<std::complex<T>> X(n), y(n);
    const std::array<std::size_t, 1> dims{n};

    auto t0 = Clock::now();
    plan.execute(Spectrum::Direction::Forward, x.data(), X.data());
    const double fwd_ms = elapsed_ms(t0);
    t0 = Clock::now();
    plan.execute(Spectrum::Direction::Inverse, X.data(), y.data());
    const double inv_ms = elapsed_ms(t0);

    const double rt = relative_max_error<T>(y, x);
    const double pv = parseval_error<T>(x, X);
    const double bins = sampled_bin_error<T, 1>(x, X, dims);

    print_metrics(label, n, fwd_ms, inv_ms, rt, pv, bins);
    FFT_STRESS_LE(rt, tol.roundtrip);
    FFT_STRESS_LE(pv, tol.parseval);
    FFT_STRESS_LE(bins, tol.bins);
}

/**
 * @brief Runs the real-to-complex API on a real signal of length @p n.
 *
 * The N/2+1 non-redundant bins are expanded with Hermitian symmetry, so the
 * same full-spectrum checks apply to them.
 */
void check_1d_real(std::size_t n, std::uint32_t seed, const Tolerances& tol, const char* label) {
    const auto xc = make_real_as_complex(n, seed);
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = xc[i].real();

    auto t0 = Clock::now();
    const auto half = Spectrum::forward_real(x);
    const double fwd_ms = elapsed_ms(t0);
    t0 = Clock::now();
    const auto back = Spectrum::inverse_real(half, n);
    const double inv_ms = elapsed_ms(t0);

    FFT_STRESS_REQUIRE(half.size() == n / 2 + 1);
    FFT_STRESS_REQUIRE(back.size() == n);

    std::vector<std::complex<double>> X(n);
    for (std::size_t k = 0; k < half.size(); ++k)
        X[k] = half[k];
    for (std::size_t k = half.size(); k < n; ++k)
        X[k] = std::conj(half[n - k]);

    std::vector<std::complex<double>> y(n);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = back[i];

    const double rt = relative_max_error<double>(y, xc);
    const double pv = parseval_error<double>(xc, X);
    const std::array<std::size_t, 1> dims{n};
    const double bins = sampled_bin_error<double, 1>(xc, X, dims);

    print_metrics(label, n, fwd_ms, inv_ms, rt, pv, bins);
    FFT_STRESS_LE(rt, tol.roundtrip);
    FFT_STRESS_LE(pv, tol.parseval);
    FFT_STRESS_LE(bins, tol.bins);
}

} // namespace

int main() {
    std::printf("Spectrum++ stress tests (large inputs)\n");

    // 1D complex, double precision: radix-4, mixed-radix, Bluestein, Rader.
    const Tolerances tol_1d_double{1e-12, 1e-12, 1e-9};
    check_1d_complex<double>(std::size_t{1} << 20, Spectrum::Algorithm::Radix4, 11u, tol_1d_double,
                             "1d complex double radix-4");
    check_1d_complex<double>(1000000, Spectrum::Algorithm::MixedRadix, 12u, tol_1d_double,
                             "1d complex double mixed-radix");
    check_1d_complex<double>(999983, Spectrum::Algorithm::Bluestein, 13u, tol_1d_double,
                             "1d complex double Bluestein");
    check_1d_complex<double>(65537, Spectrum::Algorithm::Rader, 14u, tol_1d_double,
                             "1d complex double Rader");

    // 1D complex, single precision.
    const Tolerances tol_1d_float{1e-5, 1e-5, 1e-3};
    check_1d_complex<float>(std::size_t{1} << 20, Spectrum::Algorithm::Radix4, 21u, tol_1d_float,
                            "1d complex float radix-4");
    check_1d_complex<float>(999983, Spectrum::Algorithm::Bluestein, 22u, tol_1d_float,
                            "1d complex float Bluestein");

    // 1D real-input transforms: even and odd lengths.
    check_1d_real(std::size_t{1} << 20, 31u, tol_1d_double, "1d real double even");
    check_1d_real(1000000, 32u, tol_1d_double, "1d real double even mixed");
    check_1d_real(999999, 33u, tol_1d_double, "1d real double odd");

    // 2D complex: power-of-two, rectangular, and a prime row length (Bluestein).
    check_complex<double, 2>({1024, 1024}, 41u, tol_1d_double, "2d complex double 1024x1024");
    check_complex<double, 2>({1000, 750}, 42u, tol_1d_double, "2d complex double 1000x750");
    check_complex<double, 2>({1009, 512}, 43u, tol_1d_double, "2d complex double 1009x512");
    check_complex<float, 2>({1024, 1024}, 44u, tol_1d_float, "2d complex float 1024x1024");

    // 2D real-input transforms: zero imaginary part, Hermitian spectrum.
    check_complex<double, 2>({512, 768}, 51u, tol_1d_double, "2d real double 512x768", true);

    // 3D complex: cubic and non-cubic grids.
    check_complex<double, 3>({128, 128, 128}, 61u, tol_1d_double, "3d complex double 128^3");
    check_complex<double, 3>({96, 80, 64}, 62u, tol_1d_double, "3d complex double 96x80x64");
    check_complex<float, 3>({64, 64, 64}, 63u, tol_1d_float, "3d complex float 64^3");

    // 3D real-input transform.
    check_complex<double, 3>({64, 48, 40}, 71u, tol_1d_double, "3d real double 64x48x40", true);

    if (g_failures == 0)
        std::printf("all stress checks passed\n");
    else
        std::fprintf(stderr, "%d stress check(s) failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
