#include <fft/fft.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace {

/// Number of failed checks; main() returns non-zero if any check failed.
int g_failures = 0;

/**
 * @brief Records a failed check and reports its location.
 *
 * Unlike assert(), this is not disabled by NDEBUG, so the tests still
 * fail in Release builds.
 */
void report_failure(const char* expression, const char* file, int line) {
    std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
    ++g_failures;
}

/**
 * @brief Deterministic pseudo-random values in [-1, 1) for reproducible tests.
 */
std::vector<std::complex<double>> make_input(std::size_t n, std::uint32_t seed) {
    std::vector<std::complex<double>> x(n);
    std::uint32_t state = seed;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state) / 2147483648.0 - 1.0;
    };
    for (auto& v : x)
        v = {next(), next()};
    return x;
}

/**
 * @brief Naive DFT in long double, used as an independent reference.
 * @param sign -1 for the forward transform, +1 for the inverse.
 */
std::vector<std::complex<long double>> reference_dft(
        const std::vector<std::complex<double>>& x, int sign) {
    const std::size_t n = x.size();
    std::vector<std::complex<long double>> X(n);
    const long double pi = std::numbers::pi_v<long double>;
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<long double> acc{0.0L, 0.0L};
        for (std::size_t j = 0; j < n; ++j) {
            const long double angle =
                sign * 2.0L * pi *
                static_cast<long double>((k * j) % n) / static_cast<long double>(n);
            acc += std::complex<long double>(x[j].real(), x[j].imag()) *
                   std::complex<long double>(std::cos(angle), std::sin(angle));
        }
        X[k] = acc;
    }
    return X;
}

/**
 * @brief Runs a plan of type T on double-precision input, converting in and out.
 */
template<typename T>
std::vector<std::complex<double>> run_plan(const Spectrum::FFTPlan<T>& plan,
                                           Spectrum::Direction direction,
                                           const std::vector<std::complex<double>>& x) {
    std::vector<std::complex<T>> in(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        in[i] = std::complex<T>(static_cast<T>(x[i].real()), static_cast<T>(x[i].imag()));
    std::vector<std::complex<T>> out(x.size());
    plan.execute(direction, in.data(), out.data());

    std::vector<std::complex<double>> result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        result[i] = std::complex<double>(out[i]);
    return result;
}

/**
 * @brief Largest absolute difference between @p a and a long double reference.
 */
double max_abs_error(const std::vector<std::complex<double>>& a,
                     const std::vector<std::complex<long double>>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::abs(
            std::complex<long double>(a[i].real(), a[i].imag()) - b[i])));
    return worst;
}

} // namespace

#define FFT_TEST_CHECK(cond)                                    \
    do {                                                        \
        if (!(cond)) report_failure(#cond, __FILE__, __LINE__); \
    } while (false)

/**
 * @brief Verifies a basic plan: size, precision, normalization and algorithm.
 */
static void test_plan_metadata() {
    Spectrum::FFTPlan<float> pf(256);
    FFT_TEST_CHECK(pf.size() == 256);
    FFT_TEST_CHECK(pf.precision() == Spectrum::Precision::Float32);
    FFT_TEST_CHECK(pf.normalization() == Spectrum::Normalization::Inverse);
    FFT_TEST_CHECK(pf.shape().element_count() == 256);
    FFT_TEST_CHECK(pf.algorithm() == Spectrum::Algorithm::Radix4);  // 256 = 4^4

    Spectrum::FFTPlan<double> pd(1024, Spectrum::Normalization::Ortho);
    FFT_TEST_CHECK(pd.size() == 1024);
    FFT_TEST_CHECK(pd.precision() == Spectrum::Precision::Float64);
    FFT_TEST_CHECK(pd.normalization() == Spectrum::Normalization::Ortho);
}

/**
 * @brief Verifies that invalid sizes are rejected with the proper exception.
 */
static void test_invalid_sizes() {
    bool caught = false;
    try {
        Spectrum::FFTPlan<double> invalid(0);
    } catch (const Spectrum::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);

    caught = false;
    try {
        Spectrum::Shape<2> invalid_shape({4, 0});
    } catch (const Spectrum::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);
}

/**
 * @brief Verifies that an element count overflow is detected.
 */
static void test_shape_overflow() {
    constexpr std::size_t big = std::numeric_limits<std::size_t>::max() / 2 + 1;
    Spectrum::Shape<2> shape({big, 2});

    bool caught = false;
    try {
        (void)shape.element_count();
    } catch (const Spectrum::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);
}

/**
 * @brief Verifies multi-dimensional shape metadata.
 */
static void test_shape_metadata() {
    Spectrum::Shape<2> shape2({64, 32});
    FFT_TEST_CHECK(shape2.element_count() == 2048);
    FFT_TEST_CHECK(shape2.extent(0) == 64);
    FFT_TEST_CHECK(shape2.extent(1) == 32);
}

/**
 * @brief Verifies workspace capacity and resizing.
 */
static void test_workspace() {
    Spectrum::Workspace<double> workspace(128);
    FFT_TEST_CHECK(workspace.capacity() == 128);

    workspace.resize(16);
    FFT_TEST_CHECK(workspace.capacity() == 16);
    FFT_TEST_CHECK(workspace.data() != nullptr);
}

/**
 * @brief Verifies the basic API: forward, inverse, vector and span overloads.
 */
static void test_simple_api() {
    // Impulse at 0 transforms to all ones; the vector overload is used here.
    const std::vector<std::complex<double>> impulse = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    const auto X = Spectrum::forward(impulse);
    FFT_TEST_CHECK(X.size() == 4);
    for (const auto& v : X)
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    // Inverse of [N, 0, ...] is all ones under the default normalization.
    const std::vector<std::complex<double>> dc = {{4.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    const auto x = Spectrum::inverse(dc);
    for (const auto& v : x)
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    // Round trip reproduces the input, in both precisions.
    const auto sig = make_input(12, 2024u);
    const auto back = Spectrum::inverse(Spectrum::forward(sig));
    for (std::size_t i = 0; i < sig.size(); ++i)
        FFT_TEST_CHECK(std::abs(back[i] - sig[i]) < 1e-12);

    std::vector<std::complex<float>> sigf(sig.size());
    for (std::size_t i = 0; i < sig.size(); ++i)
        sigf[i] = {static_cast<float>(sig[i].real()), static_cast<float>(sig[i].imag())};
    const auto backf = Spectrum::inverse(Spectrum::forward(sigf));
    for (std::size_t i = 0; i < sigf.size(); ++i)
        FFT_TEST_CHECK(std::abs(backf[i] - sigf[i]) < 1e-4f);

    // The span overload agrees with the vector overload.
    const auto via_span = Spectrum::forward(std::span<const std::complex<double>>(sig));
    const auto via_vec = Spectrum::forward(sig);
    for (std::size_t i = 0; i < sig.size(); ++i)
        FFT_TEST_CHECK(via_span[i] == via_vec[i]);

    // Basic API matches the advanced API with default normalization.
    Spectrum::FFTPlan<double> plan(sig.size());
    std::vector<std::complex<double>> adv(sig.size());
    plan.execute(Spectrum::Direction::Forward, sig.data(), adv.data());
    for (std::size_t i = 0; i < sig.size(); ++i)
        FFT_TEST_CHECK(std::abs(adv[i] - via_vec[i]) < 1e-12);

    // Empty input is rejected.
    bool caught = false;
    try {
        (void)Spectrum::forward(std::vector<std::complex<double>>{});
    } catch (const Spectrum::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);
}

/**
 * @brief Verifies radix-4 and radix-2 selection and accuracy on power-of-two lengths.
 */
static void test_radix_kernels() {
    // Powers of four select Radix4, other powers of two Radix2, the rest DirectDFT.
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(1).algorithm() == Spectrum::Algorithm::Radix2);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(4).algorithm() == Spectrum::Algorithm::Radix4);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(64).algorithm() == Spectrum::Algorithm::Radix4);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(128).algorithm() == Spectrum::Algorithm::Radix2);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(30).algorithm() == Spectrum::Algorithm::MixedRadix);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(7).algorithm() == Spectrum::Algorithm::DirectDFT);

    // Radix4 lengths (4, 16, 64, 256), Radix2 lengths (2, 8, 128, 512).
    const std::size_t sizes[] = {2, 4, 8, 16, 64, 128, 256, 512};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 31u + 5u);
        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        Spectrum::FFTPlan<double> pd(n, Spectrum::Normalization::None);
        Spectrum::FFTPlan<float> pf(n, Spectrum::Normalization::None);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Forward, x), ref_fwd) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Inverse, x), ref_inv) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Forward, x), ref_fwd) < 1e-4 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Inverse, x), ref_inv) < 1e-4 * static_cast<double>(n));
    }

    // Round trip and Parseval on a larger power-of-two length.
    const std::size_t big = 1024;
    const auto x = make_input(big, 4242u);
    Spectrum::FFTPlan<double> ortho(big, Spectrum::Normalization::Ortho);
    const auto X = run_plan(ortho, Spectrum::Direction::Forward, x);
    double ex = 0.0, eX = 0.0;
    for (std::size_t i = 0; i < big; ++i) {
        ex += std::norm(x[i]);
        eX += std::norm(X[i]);
    }
    FFT_TEST_CHECK(std::abs(ex - eX) < 1e-9);

    Spectrum::FFTPlan<double> plan(big);
    const auto y = run_plan(plan, Spectrum::Direction::Inverse,
                            run_plan(plan, Spectrum::Direction::Forward, x));
    for (std::size_t i = 0; i < big; ++i)
        FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-12);
}

/**
 * @brief Verifies the mixed-radix path against the reference DFT and its invariants.
 */
static void test_mixed_radix() {
    // 125, 243 and 625 exercise the radix-3 and radix-5 butterflies at several levels.
    // 74 and 122 use the odd-factor butterfly for 37 and 61; 121, 343 and 1001
    // use it for 11, 7, 7*11*13 and mix it with the closed-form factors.
    const std::size_t sizes[] = {3, 5, 6, 9, 12, 15, 20, 30, 45, 49, 60, 74, 100, 121, 122,
                                 125, 210, 243, 343, 360, 625, 1001};
    for (const std::size_t n : sizes) {
        // 3 and 5 are prime and use the direct DFT; the rest are composite.
        const bool prime = (n == 3 || n == 5);
        FFT_TEST_CHECK(Spectrum::FFTPlan<double>(n).algorithm() ==
                       (prime ? Spectrum::Algorithm::DirectDFT : Spectrum::Algorithm::MixedRadix));

        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 131u + 17u);
        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        Spectrum::FFTPlan<double> pd(n, Spectrum::Normalization::None);
        Spectrum::FFTPlan<float> pf(n, Spectrum::Normalization::None);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Forward, x), ref_fwd) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Inverse, x), ref_inv) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Forward, x), ref_fwd) < 1e-4 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Inverse, x), ref_inv) < 1e-4 * static_cast<double>(n));

        Spectrum::FFTPlan<double> plan(n);
        const auto y = run_plan(plan, Spectrum::Direction::Inverse,
                                run_plan(plan, Spectrum::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-12);
    }

    // Unit impulse gives a flat spectrum for a composite length.
    Spectrum::FFTPlan<double> six(6, Spectrum::Normalization::None);
    std::vector<std::complex<double>> imp(6, {0.0, 0.0});
    imp[0] = {1.0, 0.0};
    for (const auto& v : run_plan(six, Spectrum::Direction::Forward, imp))
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    // Ortho normalization preserves energy for a composite length.
    const std::size_t n = 360;
    const auto x = make_input(n, 360u);
    Spectrum::FFTPlan<double> ortho(n, Spectrum::Normalization::Ortho);
    const auto X = run_plan(ortho, Spectrum::Direction::Forward, x);
    double ex = 0.0, eX = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ex += std::norm(x[i]);
        eX += std::norm(X[i]);
    }
    FFT_TEST_CHECK(std::abs(ex - eX) < 1e-9);
}

/**
 * @brief Verifies the workspace overload: sizing, reuse without reallocation,
 *        growth, and agreement with the allocating overload.
 */
static void test_workspace_reuse() {
    // Only MixedRadix needs scratch; the radix and DFT paths report zero.
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(64).workspace_size() == 0);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(7).workspace_size() == 0);
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(360).workspace_size() == 360);

    const std::size_t n = 360;
    const auto x = make_input(n, 555u);
    Spectrum::FFTPlan<double> plan(n);

    // Exact-size workspace: no reallocation across calls.
    Spectrum::Workspace<double> ws(plan.workspace_size());
    FFT_TEST_CHECK(ws.capacity() == n);
    const auto* before = ws.data();
    std::vector<std::complex<double>> y1(n), y2(n), ref(n);
    plan.execute(Spectrum::Direction::Forward, x.data(), y1.data(), ws);
    plan.execute(Spectrum::Direction::Inverse, y1.data(), y2.data(), ws);
    FFT_TEST_CHECK(ws.data() == before);
    FFT_TEST_CHECK(ws.capacity() == n);

    // Same results as the allocating overload, bit for bit.
    plan.execute(Spectrum::Direction::Forward, x.data(), ref.data());
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(y1[i] == ref[i]);

    // Round trip through the workspace path.
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(std::abs(y2[i] - x[i]) < 1e-12);  // default Inverse scaling

    // Undersized workspace grows to the required size.
    Spectrum::Workspace<double> small;
    plan.execute(Spectrum::Direction::Forward, x.data(), y1.data(), small);
    FFT_TEST_CHECK(small.capacity() == n);

    // One workspace serves plans of different lengths.
    Spectrum::FFTPlan<double> other(60);
    const auto z = make_input(60, 9u);
    std::vector<std::complex<double>> zy(60), zref(60);
    other.execute(Spectrum::Direction::Forward, z.data(), zy.data(), ws);
    other.execute(Spectrum::Direction::Forward, z.data(), zref.data());
    for (std::size_t i = 0; i < 60; ++i)
        FFT_TEST_CHECK(zy[i] == zref[i]);
    FFT_TEST_CHECK(ws.capacity() == n);

    // Radix paths leave an empty workspace untouched.
    Spectrum::FFTPlan<double> radix(64);
    Spectrum::Workspace<double> unused;
    std::vector<std::complex<double>> rin(64, {1.0, 0.0}), rout(64);
    radix.execute(Spectrum::Direction::Forward, rin.data(), rout.data(), unused);
    FFT_TEST_CHECK(unused.capacity() == 0);
}

/**
 * @brief Verifies the Bluestein path: selection, accuracy against the reference
 *        DFT in both precisions, round trip, scratch sizing, and agreement of the
 *        workspace path with the allocating path.
 */
static void test_bluestein() {
    // Selection: composites with a large prime factor, and primes with p-1 not a power of two.
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(262).algorithm() == Spectrum::Algorithm::Bluestein);  // 2 * 131
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(317).algorithm() == Spectrum::Algorithm::Bluestein);  // 316 = 4 * 79
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(67).algorithm() == Spectrum::Algorithm::Bluestein);   // 66 is not a power of two
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(61).algorithm() == Spectrum::Algorithm::DirectDFT);   // prime <= 64

    // Scratch: two buffers of the power-of-two convolution length (317 -> 1024).
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(317).workspace_size() == 2048);

    const std::size_t sizes[] = {67, 127, 131, 262, 317};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 977u + 3u);
        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        Spectrum::FFTPlan<double> pd(n, Spectrum::Normalization::None);
        Spectrum::FFTPlan<float> pf(n, Spectrum::Normalization::None);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Forward, x), ref_fwd) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Inverse, x), ref_inv) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Forward, x), ref_fwd) < 1e-3 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Inverse, x), ref_inv) < 1e-3 * static_cast<double>(n));

        Spectrum::FFTPlan<double> plan(n);
        const auto y = run_plan(plan, Spectrum::Direction::Inverse,
                                run_plan(plan, Spectrum::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-11);
    }

    // Workspace path gives the same bits as the allocating overload.
    const std::size_t n = 262;
    const auto x = make_input(n, 4321u);
    Spectrum::FFTPlan<double> plan(n);
    Spectrum::Workspace<double> ws(plan.workspace_size());
    std::vector<std::complex<double>> a(n), b(n);
    plan.execute(Spectrum::Direction::Forward, x.data(), a.data(), ws);
    plan.execute(Spectrum::Direction::Forward, x.data(), b.data());
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(a[i] == b[i]);
}

/**
 * @brief Checks selected output bins against an O(N) long double sum.
 *
 * Used where a full reference DFT is too expensive (large N). Each bin k
 * is compared with sum_j x_j * exp(-2*pi*i*k*j/N) in long double.
 */
static void check_sampled_bins(const std::vector<std::complex<double>>& x,
                               const std::vector<std::complex<double>>& X,
                               const std::vector<std::size_t>& bins, double tol) {
    const std::size_t n = x.size();
    const long double pi = std::numbers::pi_v<long double>;
    for (const std::size_t k : bins) {
        std::complex<long double> acc{0.0L, 0.0L};
        for (std::size_t j = 0; j < n; ++j) {
            const long double ang = -2.0L * pi * static_cast<long double>((k * j) % n) / static_cast<long double>(n);
            acc += std::complex<long double>(x[j].real(), x[j].imag()) *
                   std::complex<long double>(std::cos(ang), std::sin(ang));
        }
        const std::complex<long double> got(X[k].real(), X[k].imag());
        FFT_TEST_CHECK(std::abs(got - acc) < tol * static_cast<double>(n));
    }
}

/**
 * @brief Verifies Rader's path on primes with p-1 a power of two: selection,
 *        accuracy, round trip, scratch sizing, and workspace agreement.
 */
static void test_rader() {
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(257).algorithm() == Spectrum::Algorithm::Rader);    // 256 = 2^8
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(65537).algorithm() == Spectrum::Algorithm::Rader);  // 65536 = 2^16
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(1009).algorithm() == Spectrum::Algorithm::Bluestein); // 1008 is not a power of two

    // Scratch: two buffers of p-1 elements.
    FFT_TEST_CHECK(Spectrum::FFTPlan<double>(257).workspace_size() == 512);

    // Full reference comparison at 257.
    {
        const std::size_t n = 257;
        const auto x = make_input(n, 613u);
        Spectrum::FFTPlan<double> pd(n, Spectrum::Normalization::None);
        Spectrum::FFTPlan<float> pf(n, Spectrum::Normalization::None);
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Forward, x), reference_dft(x, -1)) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Inverse, x), reference_dft(x, +1)) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Forward, x), reference_dft(x, -1)) < 1e-3 * static_cast<double>(n));

        Spectrum::FFTPlan<double> plan(n);
        const auto y = run_plan(plan, Spectrum::Direction::Inverse, run_plan(plan, Spectrum::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-11);
    }

    // 65537 is too large for the full reference: sample bins, check round trip and energy.
    {
        const std::size_t n = 65537;
        const auto x = make_input(n, 2024u);
        Spectrum::FFTPlan<double> plan(n, Spectrum::Normalization::None);
        const auto X = run_plan(plan, Spectrum::Direction::Forward, x);
        check_sampled_bins(x, X, {0, 1, 2, 12345, 65536}, 1e-10);

        Spectrum::FFTPlan<double> ortho(n, Spectrum::Normalization::Ortho);
        const auto Xo = run_plan(ortho, Spectrum::Direction::Forward, x);
        double ex = 0.0, eX = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            ex += std::norm(x[i]);
            eX += std::norm(Xo[i]);
        }
        FFT_TEST_CHECK(std::abs(ex - eX) < 1e-7 * ex);

        Spectrum::FFTPlan<double> inv(n);
        const auto y = run_plan(inv, Spectrum::Direction::Inverse, run_plan(inv, Spectrum::Direction::Forward, x));
        for (std::size_t i = 0; i < n; i += 97)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-11);
    }

    // Workspace path agrees with the allocating path.
    const std::size_t n = 257;
    const auto x = make_input(n, 77u);
    Spectrum::FFTPlan<double> plan(n);
    Spectrum::Workspace<double> ws(plan.workspace_size());
    std::vector<std::complex<double>> a(n), b(n);
    plan.execute(Spectrum::Direction::Inverse, x.data(), a.data(), ws);
    plan.execute(Spectrum::Direction::Inverse, x.data(), b.data());
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(a[i] == b[i]);
}

/**
 * @brief Verifies that impulses and constants give the known spectra.
 */
static void test_known_spectra() {
    // Unit impulse at n = 0 has a flat spectrum of ones.
    for (const std::size_t n : {1u, 2u, 5u, 8u}) {
        Spectrum::FFTPlan<double> plan(n, Spectrum::Normalization::None);
        std::vector<std::complex<double>> x(n, {0.0, 0.0});
        x[0] = {1.0, 0.0};
        const auto X = run_plan(plan, Spectrum::Direction::Forward, x);
        for (const auto& v : X)
            FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);
    }

    // Constant input has all energy in bin 0.
    const std::size_t n = 8;
    Spectrum::FFTPlan<double> plan(n, Spectrum::Normalization::None);
    std::vector<std::complex<double>> ones(n, {1.0, 0.0});
    const auto X = run_plan(plan, Spectrum::Direction::Forward, ones);
    FFT_TEST_CHECK(std::abs(X[0] - std::complex<double>(8.0, 0.0)) < 1e-12);
    for (std::size_t k = 1; k < n; ++k)
        FFT_TEST_CHECK(std::abs(X[k]) < 1e-12);
}

/**
 * @brief Compares forward and inverse execution against a long double reference.
 */
static void test_against_reference() {
    const std::size_t sizes[] = {1, 2, 3, 5, 7, 8, 16, 30};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 7919u + 1u);

        const double tol_d = 1e-12 * static_cast<double>(n);
        const double tol_f = 1e-4 * static_cast<double>(n);

        Spectrum::FFTPlan<double> pd(n, Spectrum::Normalization::None);
        Spectrum::FFTPlan<float> pf(n, Spectrum::Normalization::None);

        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Forward, x), ref_fwd) < tol_d);
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, Spectrum::Direction::Inverse, x), ref_inv) < tol_d);
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Forward, x), ref_fwd) < tol_f);
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, Spectrum::Direction::Inverse, x), ref_inv) < tol_f);
    }
}

/**
 * @brief Verifies that the default Inverse normalization round-trips the input.
 */
static void test_round_trip() {
    const std::size_t sizes[] = {1, 3, 8, 30};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, 42u + static_cast<std::uint32_t>(n));

        Spectrum::FFTPlan<double> pd(n);  // default: Normalization::Inverse
        const auto y = run_plan(pd, Spectrum::Direction::Inverse,
                                run_plan(pd, Spectrum::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-12);

        Spectrum::FFTPlan<float> pf(n);
        const auto z = run_plan(pf, Spectrum::Direction::Inverse,
                                run_plan(pf, Spectrum::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(z[i] - x[i]) < 1e-4);
    }
}

/**
 * @brief Verifies the scale factors of Forward and Ortho normalization.
 */
static void test_normalization_scaling() {
    const std::size_t n = 6;
    const auto x = make_input(n, 7u);
    const double dn = static_cast<double>(n);

    Spectrum::FFTPlan<double> none(n, Spectrum::Normalization::None);
    Spectrum::FFTPlan<double> fwd(n, Spectrum::Normalization::Forward);
    Spectrum::FFTPlan<double> ortho(n, Spectrum::Normalization::Ortho);

    const auto raw = run_plan(none, Spectrum::Direction::Forward, x);
    const auto scaled_fwd = run_plan(fwd, Spectrum::Direction::Forward, x);
    const auto scaled_ortho = run_plan(ortho, Spectrum::Direction::Forward, x);
    const auto raw_inv = run_plan(none, Spectrum::Direction::Inverse, x);
    const auto fwd_inv = run_plan(fwd, Spectrum::Direction::Inverse, x);

    for (std::size_t k = 0; k < n; ++k) {
        FFT_TEST_CHECK(std::abs(scaled_fwd[k] - raw[k] / dn) < 1e-12);
        FFT_TEST_CHECK(std::abs(scaled_ortho[k] - raw[k] / std::sqrt(dn)) < 1e-12);
        // Forward normalization leaves the inverse direction unscaled.
        FFT_TEST_CHECK(std::abs(fwd_inv[k] - raw_inv[k]) < 1e-12);
    }
}

/**
 * @brief Verifies that Ortho normalization preserves energy (Parseval).
 */
static void test_ortho_parseval() {
    const std::size_t n = 30;
    const auto x = make_input(n, 99u);
    Spectrum::FFTPlan<double> plan(n, Spectrum::Normalization::Ortho);
    const auto X = run_plan(plan, Spectrum::Direction::Forward, x);

    double ex = 0.0, eX = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ex += std::norm(x[i]);
        eX += std::norm(X[i]);
    }
    FFT_TEST_CHECK(std::abs(ex - eX) < 1e-10);
}

/**
 * @brief Verifies buffer validation: null pointers and overlapping ranges.
 */
static void test_buffer_validation() {
    Spectrum::FFTPlan<double> plan(4);
    std::complex<double> buf[4] = {};

    auto expect_invalid_buffer = [&](const std::complex<double>* in,
                                     std::complex<double>* out) {
        bool caught = false;
        try {
            plan.execute(Spectrum::Direction::Forward, in, out);
        } catch (const Spectrum::InvalidBufferException&) {
            caught = true;
        }
        return caught;
    };

    FFT_TEST_CHECK(expect_invalid_buffer(nullptr, buf));
    FFT_TEST_CHECK(expect_invalid_buffer(buf, nullptr));
    FFT_TEST_CHECK(expect_invalid_buffer(buf, buf));          // in-place
    FFT_TEST_CHECK(expect_invalid_buffer(buf, buf + 2));      // partial overlap
}

/**
 * @brief Verifies that a moved-from plan refuses to execute and reports
 *        Unselected, while a moved-to plan keeps working.
 */
static void test_move_semantics() {
    Spectrum::FFTPlan<double> a(4, Spectrum::Normalization::None);
    Spectrum::FFTPlan<double> b(std::move(a));

    FFT_TEST_CHECK(b.size() == 4);
    FFT_TEST_CHECK(b.algorithm() == Spectrum::Algorithm::Radix4);

    FFT_TEST_CHECK(a.size() == 0);
    FFT_TEST_CHECK(a.normalization() == Spectrum::Normalization::None);
    FFT_TEST_CHECK(a.algorithm() == Spectrum::Algorithm::Unselected);
    FFT_TEST_CHECK(a.precision() == Spectrum::Precision::Float64);

    std::complex<double> in[4] = {}, out[4] = {};
    bool caught = false;
    try {
        a.execute(Spectrum::Direction::Forward, in, out);
    } catch (const Spectrum::FFTException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);

    caught = false;
    try {
        (void)a.shape();
    } catch (const Spectrum::FFTException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);

    // A moved-from plan can be assigned to again.
    a = Spectrum::FFTPlan<double>(32);
    FFT_TEST_CHECK(a.size() == 32);

    b = std::move(a);
    FFT_TEST_CHECK(b.size() == 32);
}

/**
 * @brief Real input as a complex sequence with zero imaginary part.
 */
static std::vector<std::complex<double>> make_real_input(std::size_t n, std::uint32_t seed) {
    auto x = make_input(n, seed);
    for (auto& v : x)
        v = {v.real(), 0.0};
    return x;
}

/**
 * @brief Checks the R2C transform against the reference DFT for many lengths.
 *
 * Covers even lengths (packed path), odd lengths (complex path), and the
 * smallest lengths 1 and 2.
 */
static void test_real_forward_reference() {
    const std::size_t sizes[] = {1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 64, 100, 1009, 1024};
    for (std::size_t n : sizes) {
        const auto x = make_real_input(n, static_cast<std::uint32_t>(n) + 7u);
        std::vector<double> real_in(n);
        for (std::size_t i = 0; i < n; ++i)
            real_in[i] = x[i].real();

        const Spectrum::RealFFTPlan<double> plan(n, Spectrum::Normalization::None);
        FFT_TEST_CHECK(plan.spectrum_size() == n / 2 + 1);
        std::vector<std::complex<double>> X(plan.spectrum_size());
        plan.execute_forward(real_in.data(), X.data());

        const auto ref = reference_dft(x, -1);
        const double tol = 1e-12 * static_cast<double>(n) + 1e-14;
        FFT_TEST_CHECK(max_abs_error(X, ref) < tol);
    }
}

/**
 * @brief Single precision R2C against the reference, with a looser bound.
 */
static void test_real_forward_float() {
    const std::size_t n = 1024;
    const auto x = make_real_input(n, 99u);
    std::vector<float> real_in(n);
    for (std::size_t i = 0; i < n; ++i)
        real_in[i] = static_cast<float>(x[i].real());

    const Spectrum::RealFFTPlan<float> plan(n, Spectrum::Normalization::None);
    std::vector<std::complex<float>> Xf(plan.spectrum_size());
    plan.execute_forward(real_in.data(), Xf.data());

    std::vector<std::complex<double>> X(Xf.begin(), Xf.end());
    FFT_TEST_CHECK(max_abs_error(X, reference_dft(x, -1)) < 1e-3);
    FFT_TEST_CHECK(plan.precision() == Spectrum::Precision::Float32);
}

/**
 * @brief Round trip through the default normalization, for even and odd lengths.
 */
static void test_real_round_trip() {
    const std::size_t sizes[] = {1, 2, 3, 7, 8, 10, 33, 64, 1009, 1024, 4096};
    for (std::size_t n : sizes) {
        const auto x = make_real_input(n, static_cast<std::uint32_t>(n) * 3u + 1u);
        std::vector<double> in(n), back(n);
        for (std::size_t i = 0; i < n; ++i)
            in[i] = x[i].real();

        const Spectrum::RealFFTPlan<double> forward_plan(n);
        const Spectrum::RealFFTPlan<double> inverse_plan(n);
        std::vector<std::complex<double>> X(forward_plan.spectrum_size());
        forward_plan.execute_forward(in.data(), X.data());
        inverse_plan.execute_inverse(X.data(), back.data());

        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(back[i] - in[i]));
        FFT_TEST_CHECK(worst < 1e-12 * static_cast<double>(n) + 1e-14);
    }
}

/**
 * @brief DC and Nyquist bins of a real transform have zero imaginary part.
 */
static void test_real_hermitian_bins() {
    const std::size_t sizes[] = {2, 8, 10, 1024};
    for (std::size_t n : sizes) {
        const auto x = make_real_input(n, 5u + static_cast<std::uint32_t>(n));
        std::vector<double> in(n);
        for (std::size_t i = 0; i < n; ++i)
            in[i] = x[i].real();
        const Spectrum::RealFFTPlan<double> plan(n, Spectrum::Normalization::None);
        std::vector<std::complex<double>> X(plan.spectrum_size());
        plan.execute_forward(in.data(), X.data());
        FFT_TEST_CHECK(std::abs(X[0].imag()) < 1e-12);
        FFT_TEST_CHECK(std::abs(X[n / 2].imag()) < 1e-12 || n % 2 == 1);
    }
}

/**
 * @brief Normalization scaling: None gives N*x on a round trip, and Ortho is unitary.
 */
static void test_real_normalization() {
    const std::size_t n = 12;
    const auto x = make_real_input(n, 42u);
    std::vector<double> in(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = x[i].real();

    // None: forward is unscaled, inverse is unscaled, so the round trip gives N*x.
    {
        const Spectrum::RealFFTPlan<double> p(n, Spectrum::Normalization::None);
        std::vector<std::complex<double>> X(p.spectrum_size());
        std::vector<double> back(n);
        p.execute_forward(in.data(), X.data());
        p.execute_inverse(X.data(), back.data());
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(back[i] - static_cast<double>(n) * in[i]));
        FFT_TEST_CHECK(worst < 1e-10);
    }

    // Ortho: Parseval, using the Hermitian weights of the half spectrum.
    {
        const Spectrum::RealFFTPlan<double> p(n, Spectrum::Normalization::Ortho);
        std::vector<std::complex<double>> X(p.spectrum_size());
        p.execute_forward(in.data(), X.data());
        double time_energy = 0.0;
        for (double v : in)
            time_energy += v * v;
        double freq_energy = std::norm(X[0]) + std::norm(X[n / 2]);
        for (std::size_t k = 1; k < n / 2; ++k)
            freq_energy += 2.0 * std::norm(X[k]);
        FFT_TEST_CHECK(std::abs(freq_energy - time_energy) < 1e-10 * time_energy);
    }
}

/**
 * @brief Allocating, Workspace and raw-scratch overloads produce identical results.
 */
static void test_real_workspace_paths() {
    const std::size_t sizes[] = {8, 9, 1009};
    for (std::size_t n : sizes) {
        const auto x = make_real_input(n, 17u);
        std::vector<double> in(n);
        for (std::size_t i = 0; i < n; ++i)
            in[i] = x[i].real();
        const Spectrum::RealFFTPlan<double> plan(n);
        const std::size_t m = plan.spectrum_size();

        std::vector<std::complex<double>> a(m), b(m), c(m);
        plan.execute_forward(in.data(), a.data());

        Spectrum::Workspace<double> ws;
        plan.execute_forward(in.data(), b.data(), ws);
        FFT_TEST_CHECK(ws.capacity() >= plan.workspace_size());

        std::vector<std::complex<double>> raw(plan.workspace_size());
        plan.execute_forward(in.data(), c.data(), raw.data(), raw.size());
        FFT_TEST_CHECK(a == b);
        FFT_TEST_CHECK(a == c);

        std::vector<double> ra(n), rb(n);
        plan.execute_inverse(a.data(), ra.data());
        plan.execute_inverse(a.data(), rb.data(), ws);
        FFT_TEST_CHECK(ra == rb);
    }
}

/**
 * @brief Throws if @p f does not throw an exception of type E.
 */
template<typename E, typename F>
static bool throws_exception(F f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

/**
 * @brief Validation of sizes, buffers, scratch and the moved-from state.
 */
static void test_real_validation() {
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidSizeException>(
        [] { Spectrum::RealFFTPlan<double> p(0); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidSizeException>(
        [] { (void)Spectrum::forward_real(std::vector<double>{}); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidSizeException>(
        [] { (void)Spectrum::inverse_real(std::vector<std::complex<double>>(3), 8); })));

    const std::size_t n = 16;
    const Spectrum::RealFFTPlan<double> p(n);
    std::vector<double> in(n, 1.0);
    std::vector<std::complex<double>> out(p.spectrum_size());

    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_forward(nullptr, out.data()); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_forward(in.data(), nullptr); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_inverse(nullptr, in.data()); })));

    // Input and output that share storage overlap.
    std::vector<std::complex<double>> shared(n);
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_forward(reinterpret_cast<const double*>(shared.data()), shared.data()); })));

    // Scratch smaller than workspace_size(), null when required, or overlapping the input.
    std::vector<std::complex<double>> small(p.workspace_size() - 1);
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_forward(in.data(), out.data(), small.data(), small.size()); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_forward(in.data(), out.data(), nullptr, 0); })));
    std::vector<std::complex<double>> big(p.workspace_size());
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { p.execute_forward(reinterpret_cast<const double*>(big.data()), out.data(),
                                big.data(), big.size()); })));

    // Moved-from plans report an empty state and reject execution.
    Spectrum::RealFFTPlan<double> source(n, Spectrum::Normalization::Ortho);
    Spectrum::RealFFTPlan<double> target(std::move(source));
    FFT_TEST_CHECK(target.size() == n);
    FFT_TEST_CHECK(target.normalization() == Spectrum::Normalization::Ortho);
    FFT_TEST_CHECK(source.size() == 0);
    FFT_TEST_CHECK(source.spectrum_size() == 0);
    FFT_TEST_CHECK(source.normalization() == Spectrum::Normalization::None);
    FFT_TEST_CHECK(source.workspace_size() == 0);
    FFT_TEST_CHECK((throws_exception<Spectrum::FFTException>(
        [&] { source.execute_forward(in.data(), out.data()); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::FFTException>(
        [&] { source.execute_inverse(out.data(), in.data()); })));
}

/**
 * @brief Simple API: forward_real and inverse_real match the plan and reject bad sizes.
 */
static void test_real_simple_api() {
    const std::size_t n = 10;
    const auto x = make_real_input(n, 3u);
    std::vector<double> in(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = x[i].real();

    const auto spectrum = Spectrum::forward_real(in);
    FFT_TEST_CHECK(spectrum.size() == n / 2 + 1);
    const auto back = Spectrum::inverse_real(spectrum, n);
    FFT_TEST_CHECK(back.size() == n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(back[i] - in[i]));
    FFT_TEST_CHECK(worst < 1e-12);

    // The span overloads agree with the vector overloads.
    const auto via_span = Spectrum::forward_real(std::span<const double>(in));
    FFT_TEST_CHECK(via_span == spectrum);
}

/**
 * @brief Batch execution: sequential and threaded results equal single transforms.
 */
static void test_batch() {
    const std::size_t sizes[] = {8, 1000};
    for (std::size_t n : sizes) {
        const std::size_t count = 5;
        const std::size_t in_dist = n + 3;   // strided input
        const std::size_t out_dist = n + 2;  // strided output, still non-overlapping
        std::vector<std::complex<double>> in((count - 1) * in_dist + n);
        const auto seed_values = make_input(in.size(), 11u + static_cast<std::uint32_t>(n));
        in = seed_values;

        Spectrum::FFTPlan<double> plan(n);

        // Reference: one transform at a time with the allocating overload.
        std::vector<std::complex<double>> expected((count - 1) * out_dist + n);
        for (std::size_t b = 0; b < count; ++b) {
            std::vector<std::complex<double>> single(n);
            plan.execute(Spectrum::Direction::Forward, in.data() + b * in_dist, single.data());
            std::copy(single.begin(), single.end(), expected.begin() + static_cast<std::ptrdiff_t>(b * out_dist));
        }

        Spectrum::Workspace<double> ws;
        std::vector<std::complex<double>> seq(expected.size());
        plan.execute_batch(Spectrum::Direction::Forward, in.data(), seq.data(), count, in_dist, out_dist, ws);
        FFT_TEST_CHECK(seq == expected);

        for (unsigned threads : {1u, 2u, 0u, 7u}) {
            std::vector<std::complex<double>> par(expected.size());
            plan.execute_batch(Spectrum::Direction::Forward, in.data(), par.data(), count, in_dist, out_dist, threads);
            // The output padding between transforms is not written, so compare only the transforms.
            bool same = true;
            for (std::size_t b = 0; b < count; ++b)
                for (std::size_t k = 0; k < n; ++k)
                    same = same && par[b * out_dist + k] == expected[b * out_dist + k];
            FFT_TEST_CHECK(same);
        }
    }

    // A single transform with more threads than work.
    {
        const std::size_t n = 16;
        Spectrum::FFTPlan<double> plan(n);
        const auto in = make_input(n, 3u);
        std::vector<std::complex<double>> out(n), ref(n);
        plan.execute(Spectrum::Direction::Inverse, in.data(), ref.data());
        plan.execute_batch(Spectrum::Direction::Inverse, in.data(), out.data(), 1, n, n, 4u);
        FFT_TEST_CHECK(out == ref);
    }

    // Empty batch does nothing, even with null pointers.
    {
        Spectrum::FFTPlan<double> plan(8);
        plan.execute_batch(Spectrum::Direction::Forward, nullptr, nullptr, 0, 8, 8, 2u);
        Spectrum::Workspace<double> ws;
        plan.execute_batch(Spectrum::Direction::Forward, nullptr, nullptr, 0, 8, 8, ws);
        FFT_TEST_CHECK(true);
    }

    // Validation.
    {
        const std::size_t n = 8;
        Spectrum::FFTPlan<double> plan(n);
        std::vector<std::complex<double>> buf(4 * n);
        Spectrum::Workspace<double> ws;
        FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
            [&] { plan.execute_batch(Spectrum::Direction::Forward, nullptr, buf.data(), 2, n, n, ws); })));
        FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
            [&] { plan.execute_batch(Spectrum::Direction::Forward, buf.data(), buf.data() + 2 * n, 2, n, n - 1, ws); })));
        // Output stride too small for count > 1.
        FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
            [&] { plan.execute_batch(Spectrum::Direction::Forward, buf.data(), buf.data() + 2 * n, 2, n, n - 1, 2u); })));
        // Input extent overlaps output extent.
        FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
            [&] { plan.execute_batch(Spectrum::Direction::Forward, buf.data(), buf.data() + n, 2, n, n, ws); })));
        FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
            [&] { plan.execute_batch(Spectrum::Direction::Forward, buf.data(), buf.data() + n, 2, n, n, 2u); })));
    }

    // Moved-from plan rejects batches.
    {
        Spectrum::FFTPlan<double> source(8);
        Spectrum::FFTPlan<double> target(std::move(source));
        std::vector<std::complex<double>> buf(16);
        Spectrum::Workspace<double> ws;
        FFT_TEST_CHECK((throws_exception<Spectrum::FFTException>(
            [&] { source.execute_batch(Spectrum::Direction::Forward, buf.data(), buf.data() + 8, 1, 8, 8, ws); })));
        FFT_TEST_CHECK((throws_exception<Spectrum::FFTException>(
            [&] { source.execute_batch(Spectrum::Direction::Forward, buf.data(), buf.data() + 8, 1, 8, 8, 2u); })));
        (void)target;
    }
}

/**
 * @brief Reference multidimensional DFT in long double, independent of the separable algorithm.
 *
 * Row-major layout: the last axis varies fastest. The sign follows reference_dft.
 */
static std::vector<std::complex<long double>> reference_nd(
        const std::vector<std::complex<double>>& x,
        const std::vector<std::size_t>& dims, int sign) {
    const std::size_t total = x.size();
    const long double pi = std::numbers::pi_v<long double>;
    auto decode = [&dims](std::size_t index) {
        std::vector<std::size_t> idx(dims.size());
        for (std::size_t a = dims.size(); a-- > 0;) {
            idx[a] = index % dims[a];
            index /= dims[a];
        }
        return idx;
    };
    std::vector<std::complex<long double>> X(total);
    for (std::size_t k = 0; k < total; ++k) {
        const auto kidx = decode(k);
        std::complex<long double> acc{0.0L, 0.0L};
        for (std::size_t n = 0; n < total; ++n) {
            const auto nidx = decode(n);
            long double fraction = 0.0L;
            for (std::size_t a = 0; a < dims.size(); ++a)
                fraction += static_cast<long double>((kidx[a] * nidx[a]) % dims[a]) /
                            static_cast<long double>(dims[a]);
            const long double angle = sign * 2.0L * pi * fraction;
            acc += std::complex<long double>(x[n].real(), x[n].imag()) *
                   std::complex<long double>(std::cos(angle), std::sin(angle));
        }
        X[k] = acc;
    }
    return X;
}

/**
 * @brief Multidimensional plans against the reference DFT, both directions, unscaled.
 */
template<std::size_t Rank>
static void test_nd_reference(const std::array<std::size_t, Rank>& dims) {
    std::size_t total = 1;
    for (auto d : dims) total *= d;
    const auto x = make_input(total, static_cast<std::uint32_t>(total) + 3u);
    const std::vector<std::size_t> dv(dims.begin(), dims.end());

    const Spectrum::FFTPlanND<double, Rank> plan(Spectrum::Shape<Rank>(dims), Spectrum::Normalization::None);
    FFT_TEST_CHECK(plan.size() == total);

    std::vector<std::complex<double>> out(total);
    plan.execute(Spectrum::Direction::Forward, x.data(), out.data());
    FFT_TEST_CHECK(max_abs_error(out, reference_nd(x, dv, -1)) < 1e-11 * static_cast<double>(total));

    plan.execute(Spectrum::Direction::Inverse, x.data(), out.data());
    FFT_TEST_CHECK(max_abs_error(out, reference_nd(x, dv, +1)) < 1e-11 * static_cast<double>(total));
}

/**
 * @brief Round trip with the default normalization, and Parseval with Ortho.
 */
static void test_nd_round_trip_and_parseval() {
    const std::array<std::size_t, 2> d2[] = {{16, 15}, {64, 50}, {12, 10}};
    for (const auto& dims : d2) {
        const Spectrum::FFTPlanND<double, 2> plan{Spectrum::Shape<2>(dims)};
        const std::size_t total = dims[0] * dims[1];
        const auto x = make_input(total, 5u + static_cast<std::uint32_t>(total));
        std::vector<std::complex<double>> X(total), back(total);
        plan.execute(Spectrum::Direction::Forward, x.data(), X.data());
        plan.execute(Spectrum::Direction::Inverse, X.data(), back.data());
        double worst = 0.0;
        for (std::size_t k = 0; k < total; ++k)
            worst = std::max(worst, std::abs(back[k] - x[k]));
        FFT_TEST_CHECK(worst < 1e-12 * static_cast<double>(total));
    }

    const std::array<std::size_t, 3> d3[] = {{4, 5, 6}, {2, 3, 4}};
    for (const auto& dims : d3) {
        const Spectrum::FFTPlanND<double, 3> plan{Spectrum::Shape<3>(dims)};
        const std::size_t total = dims[0] * dims[1] * dims[2];
        const auto x = make_input(total, 9u);
        std::vector<std::complex<double>> X(total), back(total);
        plan.execute(Spectrum::Direction::Forward, x.data(), X.data());
        plan.execute(Spectrum::Direction::Inverse, X.data(), back.data());
        double worst = 0.0;
        for (std::size_t k = 0; k < total; ++k)
            worst = std::max(worst, std::abs(back[k] - x[k]));
        FFT_TEST_CHECK(worst < 1e-12 * static_cast<double>(total));
    }

    // Ortho is unitary: energy is preserved in both domains.
    const std::array<std::size_t, 2> dims{12, 10};
    const Spectrum::FFTPlanND<double, 2> ortho(Spectrum::Shape<2>(dims), Spectrum::Normalization::Ortho);
    const auto x = make_input(120, 21u);
    std::vector<std::complex<double>> X(120);
    ortho.execute(Spectrum::Direction::Forward, x.data(), X.data());
    double ex = 0.0, eX = 0.0;
    for (std::size_t k = 0; k < 120; ++k) {
        ex += std::norm(x[k]);
        eX += std::norm(X[k]);
    }
    FFT_TEST_CHECK(std::abs(eX - ex) < 1e-10 * ex);
}

/**
 * @brief Known spectra: an impulse at the origin is flat; a constant has only DC.
 */
static void test_nd_known_spectra() {
    const std::array<std::size_t, 2> dims{3, 4};
    const Spectrum::FFTPlanND<double, 2> plan(Spectrum::Shape<2>(dims), Spectrum::Normalization::None);
    std::vector<std::complex<double>> x(12, {0.0, 0.0}), X(12);
    x[0] = {1.0, 0.0};
    plan.execute(Spectrum::Direction::Forward, x.data(), X.data());
    for (const auto& v : X)
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    std::vector<std::complex<double>> ones(12, {1.0, 0.0});
    plan.execute(Spectrum::Direction::Forward, ones.data(), X.data());
    FFT_TEST_CHECK(std::abs(X[0] - std::complex<double>(12.0, 0.0)) < 1e-12);
    for (std::size_t k = 1; k < 12; ++k)
        FFT_TEST_CHECK(std::abs(X[k]) < 1e-12);
}

/**
 * @brief Rank 1 matches FFTPlan. In-place, out-of-place and workspace paths agree.
 */
static void test_nd_paths_and_rank1() {
    const std::size_t n = 12;
    const auto x = make_input(n, 4u);
    const Spectrum::FFTPlanND<double, 1> p1(Spectrum::Shape<1>({n}));
    const Spectrum::FFTPlan<double> p(n);
    std::vector<std::complex<double>> a(n), b(n);
    p1.execute(Spectrum::Direction::Forward, x.data(), a.data());
    p.execute(Spectrum::Direction::Forward, x.data(), b.data());
    FFT_TEST_CHECK(max_abs_error(a, std::vector<std::complex<long double>>(b.begin(), b.end())) < 1e-12);

    const Spectrum::FFTPlanND<double, 2> plan(Spectrum::Shape<2>({4, 6}));
    const auto y = make_input(24, 8u);
    std::vector<std::complex<double>> oop(24), inplace = y, ws_out(24), ws_inplace = y;
    Spectrum::Workspace<double> ws;
    plan.execute(Spectrum::Direction::Forward, y.data(), oop.data());
    plan.execute_inplace(Spectrum::Direction::Forward, inplace.data());
    plan.execute(Spectrum::Direction::Forward, y.data(), ws_out.data(), ws);
    plan.execute_inplace(Spectrum::Direction::Forward, ws_inplace.data(), ws);
    FFT_TEST_CHECK(oop == inplace);
    FFT_TEST_CHECK(oop == ws_out);
    FFT_TEST_CHECK(oop == ws_inplace);
}

/**
 * @brief Validation of buffers and the moved-from state.
 */
static void test_nd_validation() {
    const Spectrum::FFTPlanND<double, 2> plan(Spectrum::Shape<2>({4, 4}));
    std::vector<std::complex<double>> buf(16);
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { plan.execute(Spectrum::Direction::Forward, nullptr, buf.data()); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { plan.execute(Spectrum::Direction::Forward, buf.data(), nullptr); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { plan.execute(Spectrum::Direction::Forward, buf.data(), buf.data() + 4); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidBufferException>(
        [&] { plan.execute_inplace(Spectrum::Direction::Forward, nullptr); })));

    Spectrum::FFTPlanND<double, 2> source(Spectrum::Shape<2>({4, 4}), Spectrum::Normalization::Ortho);
    Spectrum::FFTPlanND<double, 2> target(std::move(source));
    FFT_TEST_CHECK(target.size() == 16);
    FFT_TEST_CHECK(source.size() == 0);
    FFT_TEST_CHECK(source.normalization() == Spectrum::Normalization::None);
    FFT_TEST_CHECK(source.workspace_size() == 0);
    FFT_TEST_CHECK((throws_exception<Spectrum::FFTException>(
        [&] { source.execute_inplace(Spectrum::Direction::Forward, buf.data()); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::FFTException>(
        [&] { (void)source.shape(); })));
}

/**
 * @brief Simple 2D and 3D API: round trip, equality with the plan, size checks.
 */
static void test_nd_simple_api() {
    const std::size_t rows = 6, cols = 10, depth = 4;
    const auto x2 = make_input(rows * cols, 13u);
    const auto X2 = Spectrum::forward_2d(x2, rows, cols);
    const auto back2 = Spectrum::inverse_2d(X2, rows, cols);
    double worst = 0.0;
    for (std::size_t k = 0; k < x2.size(); ++k)
        worst = std::max(worst, std::abs(back2[k] - x2[k]));
    FFT_TEST_CHECK(worst < 1e-12);

    const Spectrum::FFTPlanND<double, 2> plan(Spectrum::Shape<2>({rows, cols}));
    std::vector<std::complex<double>> ref(x2.size());
    plan.execute(Spectrum::Direction::Forward, x2.data(), ref.data());
    FFT_TEST_CHECK(X2 == ref);

    const auto x3 = make_input(rows * cols * depth, 17u);
    const auto X3 = Spectrum::forward_3d(std::span<const std::complex<double>>(x3), rows, cols, depth);
    const auto back3 = Spectrum::inverse_3d(X3, rows, cols, depth);
    worst = 0.0;
    for (std::size_t k = 0; k < x3.size(); ++k)
        worst = std::max(worst, std::abs(back3[k] - x3[k]));
    FFT_TEST_CHECK(worst < 1e-12);

    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidSizeException>(
        [&] { (void)Spectrum::forward_2d(x2, rows, cols + 1); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidSizeException>(
        [&] { (void)Spectrum::inverse_3d(X3, rows, cols, depth + 1); })));
    FFT_TEST_CHECK((throws_exception<Spectrum::InvalidSizeException>(
        [&] { (void)Spectrum::forward_2d(x2, 0, cols); })));
}

/**
 * @brief Real sinusoid with an integer number of cycles over @p n samples.
 */
static std::vector<double> real_tone(std::size_t n, double cycles, double amplitude) {
    std::vector<double> x(n);
    const double two_pi = 2.0 * std::numbers::pi;
    for (std::size_t i = 0; i < n; ++i)
        x[i] = amplitude * std::sin(two_pi * cycles * static_cast<double>(i) /
                                    static_cast<double>(n));
    return x;
}

/**
 * @brief Deterministic real values in [-1, 1) for the filter tests.
 */
static std::vector<double> real_noise(std::size_t n, std::uint32_t seed) {
    std::vector<double> x(n);
    std::uint32_t state = seed;
    for (auto& v : x) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<double>(state) / 2147483648.0 - 1.0;
    }
    return x;
}

/**
 * @brief Largest absolute difference between two real sequences of equal length.
 */
static double max_real_error(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

/**
 * @brief Two tones on exact bins (fs = N, so bin k is k Hz) are separated exactly
 *        by each of the four mask types.
 */
static void test_filter_separates_tones() {
    using FT = Spectrum::FilterType;
    constexpr std::size_t n = 1024;
    const double fs = 1024.0;
    const auto low = real_tone(n, 10.0, 1.0);
    const auto high = real_tone(n, 300.0, 0.5);
    std::vector<double> mix(n);
    for (std::size_t i = 0; i < n; ++i)
        mix[i] = low[i] + high[i];

    const double tol = 1e-10;
    FFT_TEST_CHECK(max_real_error(Spectrum::filter_real(mix, fs, FT::LowPass, 100.0), low) < tol);
    FFT_TEST_CHECK(max_real_error(Spectrum::filter_real(mix, fs, FT::HighPass, 100.0), high) < tol);
    FFT_TEST_CHECK(max_real_error(Spectrum::filter_real(mix, fs, FT::BandPass, 200.0, 400.0), high) < tol);
    FFT_TEST_CHECK(max_real_error(Spectrum::filter_real(mix, fs, FT::BandStop, 200.0, 400.0), low) < tol);
}

/**
 * @brief Complementary masks sum to the input, and a full-band low-pass is the identity.
 *        Uses an odd length and a non-integer bin frequency, so the test does not
 *        depend on exact bin alignment.
 */
static void test_filter_complementary() {
    using FT = Spectrum::FilterType;
    constexpr std::size_t n = 1001;
    const double fs = 1000.0;
    const auto x = real_noise(n, 12345u);

    const auto lp = Spectrum::filter_real(x, fs, FT::LowPass, 123.4);
    const auto hp = Spectrum::filter_real(x, fs, FT::HighPass, 123.4);
    const auto bp = Spectrum::filter_real(x, fs, FT::BandPass, 80.0, 250.0);
    const auto bs = Spectrum::filter_real(x, fs, FT::BandStop, 80.0, 250.0);

    std::vector<double> sum_lp(n), sum_bp(n);
    for (std::size_t i = 0; i < n; ++i) {
        sum_lp[i] = lp[i] + hp[i];
        sum_bp[i] = bp[i] + bs[i];
    }
    FFT_TEST_CHECK(max_real_error(sum_lp, x) < 1e-11);
    FFT_TEST_CHECK(max_real_error(sum_bp, x) < 1e-11);

    // Cutoff at Nyquist keeps every bin, so the output is the input.
    FFT_TEST_CHECK(max_real_error(Spectrum::filter_real(x, fs, FT::LowPass, fs / 2.0), x) < 1e-11);
}

/**
 * @brief The float instantiation agrees with the double result to single precision.
 */
static void test_filter_float() {
    using FT = Spectrum::FilterType;
    constexpr std::size_t n = 1001;
    const auto xd = real_noise(n, 777u);
    std::vector<float> xf(n);
    for (std::size_t i = 0; i < n; ++i)
        xf[i] = static_cast<float>(xd[i]);

    const auto ref = Spectrum::filter_real(xd, 1000.0, FT::LowPass, 123.4);
    const auto got = Spectrum::filter_real(xf, 1000.0f, FT::LowPass, 123.4f);
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        err = std::max(err, std::abs(ref[i] - static_cast<double>(got[i])));
    FFT_TEST_CHECK(err < 1e-4);
}

/**
 * @brief Invalid inputs are rejected; the cutoff_high of a non-band mask is ignored.
 */
static void test_filter_validation() {
    using FT = Spectrum::FilterType;
    const std::vector<double> x(16, 1.0);
    const std::vector<double> empty;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(empty, 8.0, FT::LowPass, 1.0)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 0.0, FT::LowPass, 1.0)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, -8.0, FT::LowPass, 1.0)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 8.0, FT::LowPass, -0.5)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 8.0, FT::LowPass, 4.5)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 8.0, FT::LowPass, nan)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 8.0, FT::BandPass, 3.0, 2.0)); }));
    FFT_TEST_CHECK(throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 8.0, FT::BandStop, 1.0, 5.0)); }));
    FFT_TEST_CHECK(!throws_exception<Spectrum::InvalidSizeException>(
        [&] { static_cast<void>(Spectrum::filter_real(x, 8.0, FT::LowPass, 1.0, -5.0)); }));
}

int main() {
    test_simple_api();
    test_radix_kernels();
    test_mixed_radix();
    test_workspace_reuse();
    test_bluestein();
    test_rader();
    test_plan_metadata();
    test_invalid_sizes();
    test_shape_overflow();
    test_shape_metadata();
    test_workspace();
    test_known_spectra();
    test_against_reference();
    test_round_trip();
    test_normalization_scaling();
    test_ortho_parseval();
    test_buffer_validation();
    test_move_semantics();
    test_real_forward_reference();
    test_real_forward_float();
    test_real_round_trip();
    test_real_hermitian_bins();
    test_real_normalization();
    test_real_workspace_paths();
    test_real_validation();
    test_real_simple_api();
    test_batch();
    test_nd_reference(std::array<std::size_t, 2>{3, 4});
    test_nd_reference(std::array<std::size_t, 2>{5, 6});
    test_nd_reference(std::array<std::size_t, 2>{8, 8});
    test_nd_reference(std::array<std::size_t, 2>{6, 10});
    test_nd_reference(std::array<std::size_t, 2>{1, 7});
    test_nd_reference(std::array<std::size_t, 2>{7, 1});
    test_nd_reference(std::array<std::size_t, 2>{16, 15});
    test_nd_reference(std::array<std::size_t, 3>{2, 3, 4});
    test_nd_reference(std::array<std::size_t, 3>{4, 5, 3});
    test_nd_reference(std::array<std::size_t, 3>{6, 1, 4});
    test_nd_reference(std::array<std::size_t, 3>{3, 3, 3});
    // Inner extents that are not multiples of the line block (8): partial blocks.
    test_nd_reference(std::array<std::size_t, 2>{9, 13});
    test_nd_reference(std::array<std::size_t, 3>{5, 3, 11});
    test_nd_reference(std::array<std::size_t, 3>{2, 9, 17});
    test_nd_round_trip_and_parseval();
    test_nd_known_spectra();
    test_nd_paths_and_rank1();
    test_nd_validation();
    test_nd_simple_api();
    test_filter_separates_tones();
    test_filter_complementary();
    test_filter_float();
    test_filter_validation();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "FFT core tests passed\n";
    return 0;
}
