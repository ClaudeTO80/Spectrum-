/**
 * @file nd_plan.cpp
 * @brief Implementation of FFTPlanND: separable transforms, one axis at a time.
 *
 * For each axis a with extent N_a, let inner be the product of the extents after
 * a and outer the product of the extents before it. The element at multi-index
 * (..., n_a, ...) is at offset o*N_a*inner + n_a*inner + i, where o and i index the
 * outer and inner positions. Each line along a has stride inner and is copied to a
 * contiguous buffer, transformed, and copied back.
 */

#include "fft/nd_plan.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include "fft/exceptions.hpp"
#include "fft/fft_plan.hpp"
#include "fft_internal.hpp"

namespace fft {

/**
 * @brief Internal state of FFTPlanND. Member order matches the constructor.
 */
template<typename T, std::size_t Rank>
struct FFTPlanND<T, Rank>::Impl {
    Shape<Rank> shape;                     ///< Extents of the grid.
    Normalization normalization;           ///< Scaling convention.
    std::size_t total;                     ///< Number of elements.
    std::size_t max_extent;                ///< Longest extent, the line buffer length.
    std::size_t max_axis_workspace;        ///< Largest workspace_size() over the axis plans.
    std::vector<FFTPlan<T>> axis_plans;    ///< One unscaled plan per axis.

    Impl(Shape<Rank> s, Normalization norm)
        : shape(s),
          normalization(norm),
          total(s.element_count()),
          max_extent(0),
          max_axis_workspace(0) {
        axis_plans.reserve(Rank);
        for (std::size_t a = 0; a < Rank; ++a) {
            axis_plans.emplace_back(shape.extent(a), Normalization::None);
            max_extent = std::max(max_extent, shape.extent(a));
            max_axis_workspace = std::max(max_axis_workspace, axis_plans.back().workspace_size());
        }
    }
};

template<typename T, std::size_t Rank>
FFTPlanND<T, Rank>::FFTPlanND(Shape<Rank> shape, Normalization normalization)
    : impl_(std::make_unique<Impl>(shape, normalization)) {}

template<typename T, std::size_t Rank>
FFTPlanND<T, Rank>::~FFTPlanND() = default;

template<typename T, std::size_t Rank>
FFTPlanND<T, Rank>::FFTPlanND(FFTPlanND&&) noexcept = default;

template<typename T, std::size_t Rank>
FFTPlanND<T, Rank>& FFTPlanND<T, Rank>::operator=(FFTPlanND&&) noexcept = default;

template<typename T, std::size_t Rank>
std::size_t FFTPlanND<T, Rank>::size() const noexcept {
    return impl_ ? impl_->total : 0;
}

template<typename T, std::size_t Rank>
Shape<Rank> FFTPlanND<T, Rank>::shape() const {
    if (!impl_)
        throw FFTException("FFTPlanND::shape() called on a moved-from plan");
    return impl_->shape;
}

template<typename T, std::size_t Rank>
Normalization FFTPlanND<T, Rank>::normalization() const noexcept {
    return impl_ ? impl_->normalization : Normalization::None;
}

template<typename T, std::size_t Rank>
Precision FFTPlanND<T, Rank>::precision() const noexcept {
    if constexpr (std::is_same_v<T, float>)
        return Precision::Float32;
    else
        return Precision::Float64;
}

template<typename T, std::size_t Rank>
std::size_t FFTPlanND<T, Rank>::workspace_size() const noexcept {
    if (!impl_)
        return 0;
    return 2 * impl_->max_extent + impl_->max_axis_workspace;
}

template<typename T, std::size_t Rank>
void FFTPlanND<T, Rank>::execute(Direction direction,
                                 const std::complex<T>* in,
                                 std::complex<T>* out) const {
    Workspace<T> workspace;
    execute(direction, in, out, workspace);
}

template<typename T, std::size_t Rank>
void FFTPlanND<T, Rank>::execute(Direction direction,
                                 const std::complex<T>* in,
                                 std::complex<T>* out,
                                 Workspace<T>& workspace) const {
    if (!impl_)
        throw FFTException("FFTPlanND::execute() called on a moved-from plan");
    if (in == nullptr || out == nullptr)
        throw InvalidBufferException("FFT input and output buffers must not be null");
    const std::size_t bytes = impl_->total * sizeof(std::complex<T>);
    if (detail::ranges_overlap(in, bytes, out, bytes))
        throw InvalidBufferException("FFT input and output buffers must not overlap");
    std::copy(in, in + impl_->total, out);
    execute_inplace(direction, out, workspace);
}

template<typename T, std::size_t Rank>
void FFTPlanND<T, Rank>::execute_inplace(Direction direction, std::complex<T>* data) const {
    Workspace<T> workspace;
    execute_inplace(direction, data, workspace);
}

template<typename T, std::size_t Rank>
void FFTPlanND<T, Rank>::execute_inplace(Direction direction,
                                         std::complex<T>* data,
                                         Workspace<T>& workspace) const {
    if (!impl_)
        throw FFTException("FFTPlanND::execute_inplace() called on a moved-from plan");
    if (data == nullptr)
        throw InvalidBufferException("FFT data buffer must not be null");

    const Impl& p = *impl_;
    const std::size_t need = workspace_size();
    if (workspace.capacity() < need)
        workspace.resize(need);
    std::complex<T>* line_in = workspace.data();
    std::complex<T>* line_out = line_in + p.max_extent;
    std::complex<T>* scratch = line_out + p.max_extent;
    const std::size_t scratch_size = need - 2 * p.max_extent;

    const auto& dims = p.shape.dimensions();
    for (std::size_t a = 0; a < Rank; ++a) {
        const std::size_t len = dims[a];
        std::size_t inner = 1;
        for (std::size_t b = a + 1; b < Rank; ++b)
            inner *= dims[b];
        std::size_t outer = 1;
        for (std::size_t b = 0; b < a; ++b)
            outer *= dims[b];

        const FFTPlan<T>& plan = p.axis_plans[a];
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t i = 0; i < inner; ++i) {
                const std::size_t base = o * len * inner + i;
                for (std::size_t j = 0; j < len; ++j)
                    line_in[j] = data[base + j * inner];
                plan.execute(direction, line_in, line_out, scratch, scratch_size);
                for (std::size_t j = 0; j < len; ++j)
                    data[base + j * inner] = line_out[j];
            }
        }
    }

    const T scale = static_cast<T>(detail::scale_factor(p.normalization, direction, p.total));
    if (scale != T(1)) {
        for (std::size_t k = 0; k < p.total; ++k)
            data[k] *= scale;
    }
}

// Explicit instantiation definitions.
template class FFTPlanND<float, 1>;
template class FFTPlanND<float, 2>;
template class FFTPlanND<float, 3>;
template class FFTPlanND<double, 1>;
template class FFTPlanND<double, 2>;
template class FFTPlanND<double, 3>;

} // namespace fft
