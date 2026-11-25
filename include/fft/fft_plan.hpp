#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include "fft_export.hpp"
#include "shape.hpp"
#include "types.hpp"

namespace fft {

/**
 * @brief Immutable 1D FFT plan metadata for a given precision.
 *
 * @tparam T Floating-point precision, either @c float or @c double.
 *
 * @note A plan moved from is left in an empty state: size() and
 *       normalization() return zero/None, algorithm() returns Unselected,
 *       and shape() throws FFTException. Such a plan may be destroyed or
 *       assigned to, but its metadata must not be relied upon.
 */
template<typename T>
class FFT_API FFTPlan {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "FFTPlan supports float and double only");

public:
    /**
     * @brief Constructs a plan for a transform of the given length.
     * @param size          Transform length; must be greater than zero.
     * @param normalization Scaling convention applied by the transform.
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

    /** @brief Selected algorithm; currently always Algorithm::Unselected. */
    [[nodiscard]] Algorithm algorithm() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit instantiation declarations. The FFT_API export attribute is on the
// class definition above; the definitions in src/fft_plan.cpp are unannotated.
extern template class FFTPlan<float>;
extern template class FFTPlan<double>;

} // namespace fft
