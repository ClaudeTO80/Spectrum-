#include <fft/fft.hpp>

#include <algorithm>
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
std::vector<std::complex<double>> run_plan(const fft::FFTPlan<T>& plan,
                                           fft::Direction direction,
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
    fft::FFTPlan<float> pf(256);
    FFT_TEST_CHECK(pf.size() == 256);
    FFT_TEST_CHECK(pf.precision() == fft::Precision::Float32);
    FFT_TEST_CHECK(pf.normalization() == fft::Normalization::Inverse);
    FFT_TEST_CHECK(pf.shape().element_count() == 256);
    FFT_TEST_CHECK(pf.algorithm() == fft::Algorithm::Radix4);  // 256 = 4^4

    fft::FFTPlan<double> pd(1024, fft::Normalization::Ortho);
    FFT_TEST_CHECK(pd.size() == 1024);
    FFT_TEST_CHECK(pd.precision() == fft::Precision::Float64);
    FFT_TEST_CHECK(pd.normalization() == fft::Normalization::Ortho);
}

/**
 * @brief Verifies that invalid sizes are rejected with the proper exception.
 */
static void test_invalid_sizes() {
    bool caught = false;
    try {
        fft::FFTPlan<double> invalid(0);
    } catch (const fft::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);

    caught = false;
    try {
        fft::Shape<2> invalid_shape({4, 0});
    } catch (const fft::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);
}

/**
 * @brief Verifies that an element count overflow is detected.
 */
static void test_shape_overflow() {
    constexpr std::size_t big = std::numeric_limits<std::size_t>::max() / 2 + 1;
    fft::Shape<2> shape({big, 2});

    bool caught = false;
    try {
        (void)shape.element_count();
    } catch (const fft::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);
}

/**
 * @brief Verifies multi-dimensional shape metadata.
 */
static void test_shape_metadata() {
    fft::Shape<2> shape2({64, 32});
    FFT_TEST_CHECK(shape2.element_count() == 2048);
    FFT_TEST_CHECK(shape2.extent(0) == 64);
    FFT_TEST_CHECK(shape2.extent(1) == 32);
}

/**
 * @brief Verifies workspace capacity and resizing.
 */
static void test_workspace() {
    fft::Workspace<double> workspace(128);
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
    const auto X = fft::forward(impulse);
    FFT_TEST_CHECK(X.size() == 4);
    for (const auto& v : X)
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    // Inverse of [N, 0, ...] is all ones under the default normalization.
    const std::vector<std::complex<double>> dc = {{4.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    const auto x = fft::inverse(dc);
    for (const auto& v : x)
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    // Round trip reproduces the input, in both precisions.
    const auto sig = make_input(12, 2024u);
    const auto back = fft::inverse(fft::forward(sig));
    for (std::size_t i = 0; i < sig.size(); ++i)
        FFT_TEST_CHECK(std::abs(back[i] - sig[i]) < 1e-12);

    std::vector<std::complex<float>> sigf(sig.size());
    for (std::size_t i = 0; i < sig.size(); ++i)
        sigf[i] = {static_cast<float>(sig[i].real()), static_cast<float>(sig[i].imag())};
    const auto backf = fft::inverse(fft::forward(sigf));
    for (std::size_t i = 0; i < sigf.size(); ++i)
        FFT_TEST_CHECK(std::abs(backf[i] - sigf[i]) < 1e-4f);

    // The span overload agrees with the vector overload.
    const auto via_span = fft::forward(std::span<const std::complex<double>>(sig));
    const auto via_vec = fft::forward(sig);
    for (std::size_t i = 0; i < sig.size(); ++i)
        FFT_TEST_CHECK(via_span[i] == via_vec[i]);

    // Basic API matches the advanced API with default normalization.
    fft::FFTPlan<double> plan(sig.size());
    std::vector<std::complex<double>> adv(sig.size());
    plan.execute(fft::Direction::Forward, sig.data(), adv.data());
    for (std::size_t i = 0; i < sig.size(); ++i)
        FFT_TEST_CHECK(std::abs(adv[i] - via_vec[i]) < 1e-12);

    // Empty input is rejected.
    bool caught = false;
    try {
        (void)fft::forward(std::vector<std::complex<double>>{});
    } catch (const fft::InvalidSizeException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);
}

/**
 * @brief Verifies radix-4 and radix-2 selection and accuracy on power-of-two lengths.
 */
static void test_radix_kernels() {
    // Powers of four select Radix4, other powers of two Radix2, the rest DirectDFT.
    FFT_TEST_CHECK(fft::FFTPlan<double>(1).algorithm() == fft::Algorithm::Radix2);
    FFT_TEST_CHECK(fft::FFTPlan<double>(4).algorithm() == fft::Algorithm::Radix4);
    FFT_TEST_CHECK(fft::FFTPlan<double>(64).algorithm() == fft::Algorithm::Radix4);
    FFT_TEST_CHECK(fft::FFTPlan<double>(128).algorithm() == fft::Algorithm::Radix2);
    FFT_TEST_CHECK(fft::FFTPlan<double>(30).algorithm() == fft::Algorithm::MixedRadix);
    FFT_TEST_CHECK(fft::FFTPlan<double>(7).algorithm() == fft::Algorithm::DirectDFT);

    // Radix4 lengths (4, 16, 64, 256), Radix2 lengths (2, 8, 128, 512).
    const std::size_t sizes[] = {2, 4, 8, 16, 64, 128, 256, 512};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 31u + 5u);
        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        fft::FFTPlan<double> pd(n, fft::Normalization::None);
        fft::FFTPlan<float> pf(n, fft::Normalization::None);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Forward, x), ref_fwd) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Inverse, x), ref_inv) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Forward, x), ref_fwd) < 1e-4 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Inverse, x), ref_inv) < 1e-4 * static_cast<double>(n));
    }

    // Round trip and Parseval on a larger power-of-two length.
    const std::size_t big = 1024;
    const auto x = make_input(big, 4242u);
    fft::FFTPlan<double> ortho(big, fft::Normalization::Ortho);
    const auto X = run_plan(ortho, fft::Direction::Forward, x);
    double ex = 0.0, eX = 0.0;
    for (std::size_t i = 0; i < big; ++i) {
        ex += std::norm(x[i]);
        eX += std::norm(X[i]);
    }
    FFT_TEST_CHECK(std::abs(ex - eX) < 1e-9);

    fft::FFTPlan<double> plan(big);
    const auto y = run_plan(plan, fft::Direction::Inverse,
                            run_plan(plan, fft::Direction::Forward, x));
    for (std::size_t i = 0; i < big; ++i)
        FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-12);
}

/**
 * @brief Verifies the mixed-radix path against the reference DFT and its invariants.
 */
static void test_mixed_radix() {
    const std::size_t sizes[] = {3, 5, 6, 9, 12, 15, 20, 30, 45, 49, 60, 100, 210, 360};
    for (const std::size_t n : sizes) {
        // 3 and 5 are prime and use the direct DFT; the rest are composite.
        const bool prime = (n == 3 || n == 5);
        FFT_TEST_CHECK(fft::FFTPlan<double>(n).algorithm() ==
                       (prime ? fft::Algorithm::DirectDFT : fft::Algorithm::MixedRadix));

        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 131u + 17u);
        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        fft::FFTPlan<double> pd(n, fft::Normalization::None);
        fft::FFTPlan<float> pf(n, fft::Normalization::None);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Forward, x), ref_fwd) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Inverse, x), ref_inv) < 1e-11 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Forward, x), ref_fwd) < 1e-4 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Inverse, x), ref_inv) < 1e-4 * static_cast<double>(n));

        fft::FFTPlan<double> plan(n);
        const auto y = run_plan(plan, fft::Direction::Inverse,
                                run_plan(plan, fft::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-12);
    }

    // Unit impulse gives a flat spectrum for a composite length.
    fft::FFTPlan<double> six(6, fft::Normalization::None);
    std::vector<std::complex<double>> imp(6, {0.0, 0.0});
    imp[0] = {1.0, 0.0};
    for (const auto& v : run_plan(six, fft::Direction::Forward, imp))
        FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);

    // Ortho normalization preserves energy for a composite length.
    const std::size_t n = 360;
    const auto x = make_input(n, 360u);
    fft::FFTPlan<double> ortho(n, fft::Normalization::Ortho);
    const auto X = run_plan(ortho, fft::Direction::Forward, x);
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
    FFT_TEST_CHECK(fft::FFTPlan<double>(64).workspace_size() == 0);
    FFT_TEST_CHECK(fft::FFTPlan<double>(7).workspace_size() == 0);
    FFT_TEST_CHECK(fft::FFTPlan<double>(360).workspace_size() == 360);

    const std::size_t n = 360;
    const auto x = make_input(n, 555u);
    fft::FFTPlan<double> plan(n);

    // Exact-size workspace: no reallocation across calls.
    fft::Workspace<double> ws(plan.workspace_size());
    FFT_TEST_CHECK(ws.capacity() == n);
    const auto* before = ws.data();
    std::vector<std::complex<double>> y1(n), y2(n), ref(n);
    plan.execute(fft::Direction::Forward, x.data(), y1.data(), ws);
    plan.execute(fft::Direction::Inverse, y1.data(), y2.data(), ws);
    FFT_TEST_CHECK(ws.data() == before);
    FFT_TEST_CHECK(ws.capacity() == n);

    // Same results as the allocating overload, bit for bit.
    plan.execute(fft::Direction::Forward, x.data(), ref.data());
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(y1[i] == ref[i]);

    // Round trip through the workspace path.
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(std::abs(y2[i] - x[i]) < 1e-12);  // default Inverse scaling

    // Undersized workspace grows to the required size.
    fft::Workspace<double> small;
    plan.execute(fft::Direction::Forward, x.data(), y1.data(), small);
    FFT_TEST_CHECK(small.capacity() == n);

    // One workspace serves plans of different lengths.
    fft::FFTPlan<double> other(60);
    const auto z = make_input(60, 9u);
    std::vector<std::complex<double>> zy(60), zref(60);
    other.execute(fft::Direction::Forward, z.data(), zy.data(), ws);
    other.execute(fft::Direction::Forward, z.data(), zref.data());
    for (std::size_t i = 0; i < 60; ++i)
        FFT_TEST_CHECK(zy[i] == zref[i]);
    FFT_TEST_CHECK(ws.capacity() == n);

    // Radix paths leave an empty workspace untouched.
    fft::FFTPlan<double> radix(64);
    fft::Workspace<double> unused;
    std::vector<std::complex<double>> rin(64, {1.0, 0.0}), rout(64);
    radix.execute(fft::Direction::Forward, rin.data(), rout.data(), unused);
    FFT_TEST_CHECK(unused.capacity() == 0);
}

/**
 * @brief Verifies the Bluestein path: selection, accuracy against the reference
 *        DFT in both precisions, round trip, scratch sizing, and agreement of the
 *        workspace path with the allocating path.
 */
static void test_bluestein() {
    // Selection: composites with a large prime factor, and primes with p-1 not a power of two.
    FFT_TEST_CHECK(fft::FFTPlan<double>(262).algorithm() == fft::Algorithm::Bluestein);  // 2 * 131
    FFT_TEST_CHECK(fft::FFTPlan<double>(317).algorithm() == fft::Algorithm::Bluestein);  // 316 = 4 * 79
    FFT_TEST_CHECK(fft::FFTPlan<double>(67).algorithm() == fft::Algorithm::Bluestein);   // 66 is not a power of two
    FFT_TEST_CHECK(fft::FFTPlan<double>(61).algorithm() == fft::Algorithm::DirectDFT);   // prime <= 64

    // Scratch: two buffers of the power-of-two convolution length (317 -> 1024).
    FFT_TEST_CHECK(fft::FFTPlan<double>(317).workspace_size() == 2048);

    const std::size_t sizes[] = {67, 127, 131, 262, 317};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, static_cast<std::uint32_t>(n) * 977u + 3u);
        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        fft::FFTPlan<double> pd(n, fft::Normalization::None);
        fft::FFTPlan<float> pf(n, fft::Normalization::None);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Forward, x), ref_fwd) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Inverse, x), ref_inv) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Forward, x), ref_fwd) < 1e-3 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Inverse, x), ref_inv) < 1e-3 * static_cast<double>(n));

        fft::FFTPlan<double> plan(n);
        const auto y = run_plan(plan, fft::Direction::Inverse,
                                run_plan(plan, fft::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-11);
    }

    // Workspace path gives the same bits as the allocating overload.
    const std::size_t n = 262;
    const auto x = make_input(n, 4321u);
    fft::FFTPlan<double> plan(n);
    fft::Workspace<double> ws(plan.workspace_size());
    std::vector<std::complex<double>> a(n), b(n);
    plan.execute(fft::Direction::Forward, x.data(), a.data(), ws);
    plan.execute(fft::Direction::Forward, x.data(), b.data());
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
    FFT_TEST_CHECK(fft::FFTPlan<double>(257).algorithm() == fft::Algorithm::Rader);    // 256 = 2^8
    FFT_TEST_CHECK(fft::FFTPlan<double>(65537).algorithm() == fft::Algorithm::Rader);  // 65536 = 2^16
    FFT_TEST_CHECK(fft::FFTPlan<double>(1009).algorithm() == fft::Algorithm::Bluestein); // 1008 is not a power of two

    // Scratch: two buffers of p-1 elements.
    FFT_TEST_CHECK(fft::FFTPlan<double>(257).workspace_size() == 512);

    // Full reference comparison at 257.
    {
        const std::size_t n = 257;
        const auto x = make_input(n, 613u);
        fft::FFTPlan<double> pd(n, fft::Normalization::None);
        fft::FFTPlan<float> pf(n, fft::Normalization::None);
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Forward, x), reference_dft(x, -1)) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Inverse, x), reference_dft(x, +1)) < 1e-10 * static_cast<double>(n));
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Forward, x), reference_dft(x, -1)) < 1e-3 * static_cast<double>(n));

        fft::FFTPlan<double> plan(n);
        const auto y = run_plan(plan, fft::Direction::Inverse, run_plan(plan, fft::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-11);
    }

    // 65537 is too large for the full reference: sample bins, check round trip and energy.
    {
        const std::size_t n = 65537;
        const auto x = make_input(n, 2024u);
        fft::FFTPlan<double> plan(n, fft::Normalization::None);
        const auto X = run_plan(plan, fft::Direction::Forward, x);
        check_sampled_bins(x, X, {0, 1, 2, 12345, 65536}, 1e-10);

        fft::FFTPlan<double> ortho(n, fft::Normalization::Ortho);
        const auto Xo = run_plan(ortho, fft::Direction::Forward, x);
        double ex = 0.0, eX = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            ex += std::norm(x[i]);
            eX += std::norm(Xo[i]);
        }
        FFT_TEST_CHECK(std::abs(ex - eX) < 1e-7 * ex);

        fft::FFTPlan<double> inv(n);
        const auto y = run_plan(inv, fft::Direction::Inverse, run_plan(inv, fft::Direction::Forward, x));
        for (std::size_t i = 0; i < n; i += 97)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-11);
    }

    // Workspace path agrees with the allocating path.
    const std::size_t n = 257;
    const auto x = make_input(n, 77u);
    fft::FFTPlan<double> plan(n);
    fft::Workspace<double> ws(plan.workspace_size());
    std::vector<std::complex<double>> a(n), b(n);
    plan.execute(fft::Direction::Inverse, x.data(), a.data(), ws);
    plan.execute(fft::Direction::Inverse, x.data(), b.data());
    for (std::size_t i = 0; i < n; ++i)
        FFT_TEST_CHECK(a[i] == b[i]);
}

/**
 * @brief Verifies that impulses and constants give the known spectra.
 */
static void test_known_spectra() {
    // Unit impulse at n = 0 has a flat spectrum of ones.
    for (const std::size_t n : {1u, 2u, 5u, 8u}) {
        fft::FFTPlan<double> plan(n, fft::Normalization::None);
        std::vector<std::complex<double>> x(n, {0.0, 0.0});
        x[0] = {1.0, 0.0};
        const auto X = run_plan(plan, fft::Direction::Forward, x);
        for (const auto& v : X)
            FFT_TEST_CHECK(std::abs(v - std::complex<double>(1.0, 0.0)) < 1e-12);
    }

    // Constant input has all energy in bin 0.
    const std::size_t n = 8;
    fft::FFTPlan<double> plan(n, fft::Normalization::None);
    std::vector<std::complex<double>> ones(n, {1.0, 0.0});
    const auto X = run_plan(plan, fft::Direction::Forward, ones);
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

        fft::FFTPlan<double> pd(n, fft::Normalization::None);
        fft::FFTPlan<float> pf(n, fft::Normalization::None);

        const auto ref_fwd = reference_dft(x, -1);
        const auto ref_inv = reference_dft(x, +1);

        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Forward, x), ref_fwd) < tol_d);
        FFT_TEST_CHECK(max_abs_error(run_plan(pd, fft::Direction::Inverse, x), ref_inv) < tol_d);
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Forward, x), ref_fwd) < tol_f);
        FFT_TEST_CHECK(max_abs_error(run_plan(pf, fft::Direction::Inverse, x), ref_inv) < tol_f);
    }
}

/**
 * @brief Verifies that the default Inverse normalization round-trips the input.
 */
static void test_round_trip() {
    const std::size_t sizes[] = {1, 3, 8, 30};
    for (const std::size_t n : sizes) {
        const auto x = make_input(n, 42u + static_cast<std::uint32_t>(n));

        fft::FFTPlan<double> pd(n);  // default: Normalization::Inverse
        const auto y = run_plan(pd, fft::Direction::Inverse,
                                run_plan(pd, fft::Direction::Forward, x));
        for (std::size_t i = 0; i < n; ++i)
            FFT_TEST_CHECK(std::abs(y[i] - x[i]) < 1e-12);

        fft::FFTPlan<float> pf(n);
        const auto z = run_plan(pf, fft::Direction::Inverse,
                                run_plan(pf, fft::Direction::Forward, x));
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

    fft::FFTPlan<double> none(n, fft::Normalization::None);
    fft::FFTPlan<double> fwd(n, fft::Normalization::Forward);
    fft::FFTPlan<double> ortho(n, fft::Normalization::Ortho);

    const auto raw = run_plan(none, fft::Direction::Forward, x);
    const auto scaled_fwd = run_plan(fwd, fft::Direction::Forward, x);
    const auto scaled_ortho = run_plan(ortho, fft::Direction::Forward, x);
    const auto raw_inv = run_plan(none, fft::Direction::Inverse, x);
    const auto fwd_inv = run_plan(fwd, fft::Direction::Inverse, x);

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
    fft::FFTPlan<double> plan(n, fft::Normalization::Ortho);
    const auto X = run_plan(plan, fft::Direction::Forward, x);

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
    fft::FFTPlan<double> plan(4);
    std::complex<double> buf[4] = {};

    auto expect_invalid_buffer = [&](const std::complex<double>* in,
                                     std::complex<double>* out) {
        bool caught = false;
        try {
            plan.execute(fft::Direction::Forward, in, out);
        } catch (const fft::InvalidBufferException&) {
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
    fft::FFTPlan<double> a(4, fft::Normalization::None);
    fft::FFTPlan<double> b(std::move(a));

    FFT_TEST_CHECK(b.size() == 4);
    FFT_TEST_CHECK(b.algorithm() == fft::Algorithm::Radix4);

    FFT_TEST_CHECK(a.size() == 0);
    FFT_TEST_CHECK(a.normalization() == fft::Normalization::None);
    FFT_TEST_CHECK(a.algorithm() == fft::Algorithm::Unselected);
    FFT_TEST_CHECK(a.precision() == fft::Precision::Float64);

    std::complex<double> in[4] = {}, out[4] = {};
    bool caught = false;
    try {
        a.execute(fft::Direction::Forward, in, out);
    } catch (const fft::FFTException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);

    caught = false;
    try {
        (void)a.shape();
    } catch (const fft::FFTException&) {
        caught = true;
    }
    FFT_TEST_CHECK(caught);

    // A moved-from plan can be assigned to again.
    a = fft::FFTPlan<double>(32);
    FFT_TEST_CHECK(a.size() == 32);

    b = std::move(a);
    FFT_TEST_CHECK(b.size() == 32);
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

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "FFT core tests passed\n";
    return 0;
}
