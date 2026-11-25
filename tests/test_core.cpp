#include <fft/fft.hpp>

#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>

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

} // namespace

#define FFT_TEST_CHECK(cond)                                  \
    do {                                                      \
        if (!(cond)) report_failure(#cond, __FILE__, __LINE__); \
    } while (false)

/**
 * @brief Verifies a basic plan: size, precision, normalization and shape.
 */
static void test_plan_metadata() {
    fft::FFTPlan<float> pf(256);
    FFT_TEST_CHECK(pf.size() == 256);
    FFT_TEST_CHECK(pf.precision() == fft::Precision::Float32);
    FFT_TEST_CHECK(pf.normalization() == fft::Normalization::Inverse);
    FFT_TEST_CHECK(pf.shape().element_count() == 256);
    FFT_TEST_CHECK(pf.algorithm() == fft::Algorithm::Unselected);

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
 * @brief Verifies move construction and assignment, including the
 *        moved-from state documented in fft_plan.hpp.
 */
static void test_move_semantics() {
    fft::FFTPlan<double> a(64, fft::Normalization::None);
    fft::FFTPlan<double> b(std::move(a));

    FFT_TEST_CHECK(b.size() == 64);
    FFT_TEST_CHECK(b.normalization() == fft::Normalization::None);

    // Moved-from state: metadata accessors must not dereference null.
    FFT_TEST_CHECK(a.size() == 0);
    FFT_TEST_CHECK(a.normalization() == fft::Normalization::None);
    FFT_TEST_CHECK(a.precision() == fft::Precision::Float64);

    bool caught = false;
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
    test_plan_metadata();
    test_invalid_sizes();
    test_shape_overflow();
    test_shape_metadata();
    test_workspace();
    test_move_semantics();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "FFT core tests passed\n";
    return 0;
}
