#pragma once

/**
 * @file real_plan.hpp
 * @brief Plan for transforms of real-valued sequences (real-to-complex and back).
 */

#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "fft_export.hpp"
#include "types.hpp"
#include "workspace.hpp"

namespace Spectrum {

/**
 * @brief Immutable plan for the real-to-complex (R2C) and complex-to-real (C2R)
 *        transforms of a sequence of length N.
 *
 * A real sequence of length N has a Hermitian spectrum, X[N-k] = conj(X[k]), so
 * only the N/2+1 bins X[0] .. X[N/2] are stored and returned. The spectrum of a
 * real input is therefore of size spectrum_size() = N/2 + 1 (integer division).
 *
 * For even N the plan packs the real input into N/2 complex samples, runs one
 * complex FFT of length N/2 and then recovers the spectrum with a post-processing
 * pass. This costs about half of a complex transform of length N. For odd N it
 * runs a complex FFT of length N, which is correct but gives no speed-up.
 *
 * Sign convention and normalization are the same as FFTPlan: Forward uses
 * exp(-2*pi*i*k*n/N), and the scale factor follows the Normalization given to the
 * constructor. The default is Normalization::Inverse, so inverse(forward(x)) = x.
 *
 * The plan is immutable: execution does not change it, so one plan may be shared
 * by concurrent callers as long as each call uses its own buffers or workspace.
 *
 * @tparam T Floating-point precision, either @c float or @c double.
 *
 * @note A plan moved from is left in an empty state: size() and spectrum_size()
 *       return 0, and both execute functions throw FFTException.
 */
template<typename T>
class FFT_API RealFFTPlan {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "RealFFTPlan supports float and double only");

public:
    /**
     * @brief Constructs a plan for a real transform of length @p size.
     * @param size          Real sequence length; must be greater than zero.
     * @param normalization Scaling convention applied by the execute functions.
     * @throws InvalidSizeException if @p size is zero.
     */
    explicit RealFFTPlan(std::size_t size,
                         Normalization normalization = Normalization::Inverse);

    ~RealFFTPlan();

    RealFFTPlan(RealFFTPlan&&) noexcept;
    RealFFTPlan& operator=(RealFFTPlan&&) noexcept;

    RealFFTPlan(const RealFFTPlan&) = delete;
    RealFFTPlan& operator=(const RealFFTPlan&) = delete;

    /** @brief Length of the real sequence N, or 0 if the plan was moved from. */
    [[nodiscard]] std::size_t size() const noexcept;

    /** @brief Number of spectrum bins, N/2 + 1, or 0 if the plan was moved from. */
    [[nodiscard]] std::size_t spectrum_size() const noexcept;

    /** @brief Scaling convention, or Normalization::None if moved from. */
    [[nodiscard]] Normalization normalization() const noexcept;

    /** @brief Floating-point precision of the plan type @c T. */
    [[nodiscard]] Precision precision() const noexcept;

    /**
     * @brief Scratch elements needed by the workspace overloads.
     *
     * Two buffers of size()/2 (even N) or size() (odd N), plus the scratch of
     * the inner complex plan.
     */
    [[nodiscard]] std::size_t workspace_size() const noexcept;

    /**
     * @brief Forward real-to-complex transform.
     *
     * @param in  size() real samples. Must not be null.
     * @param out spectrum_size() bins. Must not be null and must not overlap @p in.
     * @throws InvalidBufferException if a pointer is null or the buffers overlap.
     * @throws FFTException if the plan was moved from.
     *
     * @note This overload allocates a temporary workspace on each call.
     */
    void execute_forward(const T* in, std::complex<T>* out) const;

    /**
     * @brief Forward transform with a reusable workspace (grown if too small).
     *
     * A workspace must not be used by two calls at the same time.
     */
    void execute_forward(const T* in, std::complex<T>* out,
                         Workspace<T>& workspace) const;

    /**
     * @brief Forward transform with scratch memory given as a raw buffer.
     *
     * @param scratch      At least @p scratch_size elements, and at least
     *                     workspace_size(). Must not overlap @p in or @p out.
     * @param scratch_size Number of elements in @p scratch.
     * @throws InvalidBufferException if a pointer is null, any two buffers
     *         overlap, or the scratch is too small.
     */
    void execute_forward(const T* in, std::complex<T>* out,
                         std::complex<T>* scratch, std::size_t scratch_size) const;

    /**
     * @brief Inverse complex-to-real transform.
     *
     * The input is assumed to be Hermitian. The imaginary parts of @p in[0] and
     * @p in[N/2] (for even N) are ignored, because a real sequence has real DC and
     * Nyquist bins.
     *
     * @param in  spectrum_size() bins. Must not be null.
     * @param out size() real samples. Must not be null and must not overlap @p in.
     * @throws InvalidBufferException if a pointer is null or the buffers overlap.
     * @throws FFTException if the plan was moved from.
     */
    void execute_inverse(const std::complex<T>* in, T* out) const;

    /** @brief Inverse transform with a reusable workspace (grown if too small). */
    void execute_inverse(const std::complex<T>* in, T* out,
                         Workspace<T>& workspace) const;

    /**
     * @brief Inverse transform with scratch memory given as a raw buffer.
     *
     * @throws InvalidBufferException if a pointer is null, any two buffers
     *         overlap, or the scratch is too small.
     */
    void execute_inverse(const std::complex<T>* in, T* out,
                         std::complex<T>* scratch, std::size_t scratch_size) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit instantiation declarations; the definitions are in src/real_plan.cpp.
extern template class RealFFTPlan<float>;
extern template class RealFFTPlan<double>;

} // namespace Spectrum
