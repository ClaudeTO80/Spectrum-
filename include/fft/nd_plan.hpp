#pragma once

/**
 * @file nd_plan.hpp
 * @brief Plan for multidimensional complex transforms of rank 1, 2 and 3 (and others).
 */

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
 * @brief Immutable plan for complex transforms over a row-major grid of rank Rank.
 *
 * Elements are stored row-major: the last axis varies fastest. The transform is
 * separable, so it is computed as one 1D transform along each axis, using a plan
 * for each axis length. The normalization is applied once, with the total number
 * of elements as N, so the scaling matches the 1D convention: Inverse divides by
 * the element count on the inverse, and Ortho divides by its square root on both.
 *
 * Sign convention and normalization are the same as FFTPlan. The plan is immutable
 * and may be shared between threads, provided each call uses its own buffers.
 *
 * @tparam T    Floating-point precision, @c float or @c double.
 * @tparam Rank Number of axes, at least one.
 *
 * @note A plan moved from is left empty: size() returns 0, and the execute
 *       functions and shape() throw FFTException.
 */
template<typename T, std::size_t Rank>
class FFT_API FFTPlanND {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "FFTPlanND supports float and double only");
    static_assert(Rank > 0, "FFTPlanND rank must be positive");

public:
    /**
     * @brief Constructs a plan for a grid of the given shape.
     * @param shape         Extent of each axis; every extent must be greater than zero.
     * @param normalization Scaling convention applied by the execute functions.
     * @throws InvalidSizeException if an extent is zero or the element count overflows.
     */
    explicit FFTPlanND(Shape<Rank> shape,
                       Normalization normalization = Normalization::Inverse);

    ~FFTPlanND();

    FFTPlanND(FFTPlanND&&) noexcept;
    FFTPlanND& operator=(FFTPlanND&&) noexcept;

    FFTPlanND(const FFTPlanND&) = delete;
    FFTPlanND& operator=(const FFTPlanND&) = delete;

    /** @brief Number of elements, the product of the extents; 0 if moved from. */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Shape of the transform.
     * @throws FFTException if the plan was moved from.
     */
    [[nodiscard]] Shape<Rank> shape() const;

    /** @brief Scaling convention, or Normalization::None if moved from. */
    [[nodiscard]] Normalization normalization() const noexcept;

    /** @brief Floating-point precision of the plan type @c T. */
    [[nodiscard]] Precision precision() const noexcept;

    /**
     * @brief Scratch elements that the workspace overloads need.
     *
     * Two line buffers of the longest extent, plus the largest per-axis scratch.
     */
    [[nodiscard]] std::size_t workspace_size() const noexcept;

    /**
     * @brief Out-of-place transform. Copies the input to the output, then transforms in place.
     *
     * @throws InvalidBufferException if a pointer is null or the buffers overlap.
     * @throws FFTException if the plan was moved from.
     */
    void execute(Direction direction,
                 const std::complex<T>* in,
                 std::complex<T>* out) const;

    /** @brief Out-of-place transform with a reusable workspace (grown if too small). */
    void execute(Direction direction,
                 const std::complex<T>* in,
                 std::complex<T>* out,
                 Workspace<T>& workspace) const;

    /**
     * @brief In-place transform of size() elements.
     * @throws InvalidBufferException if @p data is null.
     * @throws FFTException if the plan was moved from.
     */
    void execute_inplace(Direction direction, std::complex<T>* data) const;

    /** @brief In-place transform with a reusable workspace (grown if too small). */
    void execute_inplace(Direction direction,
                         std::complex<T>* data,
                         Workspace<T>& workspace) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit instantiation declarations for the supported ranks. The definitions
// are in src/nd_plan.cpp.
extern template class FFTPlanND<float, 1>;
extern template class FFTPlanND<float, 2>;
extern template class FFTPlanND<float, 3>;
extern template class FFTPlanND<double, 1>;
extern template class FFTPlanND<double, 2>;
extern template class FFTPlanND<double, 3>;

} // namespace fft
