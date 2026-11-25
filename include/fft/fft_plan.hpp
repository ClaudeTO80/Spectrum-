#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "fft_export.hpp"
#include "shape.hpp"
#include "types.hpp"
#include "workspace.hpp"

namespace fft {

/**
 * @brief Immutable 1D complex FFT plan for a given precision.
 *
 * The plan selects an algorithm at construction time:
 *  - Radix4 (iterative Cooley-Tukey) for powers of four;
 *  - Radix2 for other powers of two;
 *  - Bluestein (chirp-z) for primes above 64 and for composites with a prime
 *    factor above 64;
 *  - MixedRadix (recursive Cooley-Tukey over the prime factors) for other
 *    composite lengths;
 *  - DirectDFT for the remaining primes.
 * It can then execute transforms of its length any number of times. Execution does not modify
 * the plan, so a single plan may be shared by concurrent callers.
 *
 * Sign convention: Forward uses exp(-2*pi*i*k*n/N), Inverse uses
 * exp(+2*pi*i*k*n/N). Scaling per Normalization:
 *  - None:    no scaling in either direction;
 *  - Forward: 1/N on Forward only;
 *  - Inverse: 1/N on Inverse only (default);
 *  - Ortho:   1/sqrt(N) in both directions (unitary transform).
 *
 * @tparam T Floating-point precision, either @c float or @c double.
 *
 * @note A plan moved from is left in an empty state: size() returns 0,
 *       normalization() returns None, algorithm() returns Unselected,
 *       shape() and execute() throw FFTException.
 */
template<typename T>
class FFT_API FFTPlan {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "FFTPlan supports float and double only");

public:
    /**
     * @brief Constructs a plan for a transform of the given length.
     * @param size          Transform length; must be greater than zero.
     * @param normalization Scaling convention applied by execute().
     * @throws InvalidSizeException if @p size is zero.
     */
    explicit FFTPlan(std::size_t size,
                     Normalization normalization = Normalization::Inverse);

    ~FFTPlan();

    FFTPlan(FFTPlan&&) noexcept;
    FFTPlan& operator=(FFTPlan&&) noexcept;

    FFTPlan(const FFTPlan&) = delete;
    FFTPlan& operator=(const FFTPlan&) = delete;

    /** @brief Transform length, or 0 if the plan was moved from. */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief One-dimensional shape of the transform.
     * @throws FFTException if the plan was moved from.
     */
    [[nodiscard]] Shape<1> shape() const;

    /** @brief Scaling convention, or Normalization::None if moved from. */
    [[nodiscard]] Normalization normalization() const noexcept;

    /** @brief Floating-point precision of the plan type @c T. */
    [[nodiscard]] Precision precision() const noexcept;

    /** @brief Algorithm chosen at construction; Unselected if moved from. */
    [[nodiscard]] Algorithm algorithm() const noexcept;

    /**
     * @brief Scratch elements that execute(..., Workspace&) needs for this plan.
     *
     * Zero for the radix and direct paths. size() for MixedRadix, and twice the
     * convolution length for Bluestein. A Workspace of this capacity, reused
     * across calls, makes execution allocation-free.
     */
    [[nodiscard]] std::size_t workspace_size() const noexcept;

    /**
     * @brief Executes the transform on @c size() complex elements.
     *
     * @param direction Forward or Inverse.
     * @param in        Input buffer of size() elements. Must not be null.
     * @param out       Output buffer of size() elements. Must not be null and
     *                  must not overlap @p in.
     * @throws InvalidBufferException if a pointer is null or the buffers overlap.
     * @throws FFTException if the plan was moved from.
     *
     * @note This overload creates a temporary workspace. For MixedRadix that
     *       allocates size() elements on every call; use the overload below
     *       to reuse a workspace.
     */
    void execute(Direction direction,
                 const std::complex<T>* in,
                 std::complex<T>* out) const;

    /**
     * @brief Executes the transform using caller-owned scratch memory.
     *
     * Behaves as the three-argument overload. The workspace is grown to
     * workspace_size() if its capacity is smaller, and is otherwise used as is,
     * so repeated calls with the same workspace do not allocate. A workspace
     * must not be used by two calls at the same time.
     *
     * @param direction Forward or Inverse.
     * @param in        Input buffer of size() elements. Must not be null.
     * @param out       Output buffer of size() elements. Must not be null and
     *                  must not overlap @p in.
     * @param workspace Scratch memory, reused and possibly grown.
     * @throws InvalidBufferException if a pointer is null or the buffers overlap.
     * @throws FFTException if the plan was moved from.
     */
    void execute(Direction direction,
                 const std::complex<T>* in,
                 std::complex<T>* out,
                 Workspace<T>& workspace) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit instantiation declarations. The FFT_API export attribute is on the
// class definition above; the definitions in src/fft_plan.cpp are unannotated.
extern template class FFTPlan<float>;
extern template class FFTPlan<double>;

} // namespace fft
