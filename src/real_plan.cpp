/**
 * @file real_plan.cpp
 * @brief Implementation of RealFFTPlan: packed even-length and direct odd-length
 *        real-to-complex and complex-to-real transforms.
 *
 * Notation: N is the real length, m = N/2 for even N, and W^k = exp(-2*pi*i*k/N).
 * Forward even case: the real input x is packed as z[j] = x[2j] + i*x[2j+1], and
 * Z = DFT_m(z). The spectrum is X[k] = E[k] + W^k * O[k], where
 * E[k] = (Z[k] + conj(Z[m-k])) / 2 and O[k] = (Z[k] - conj(Z[m-k])) / (2i).
 * Inverse even case: the process is reversed, and the inner unscaled inverse
 * returns m * (x_even + i * x_odd), so the outputs are 2 * Re and 2 * Im of it.
 */

#include "fft/real_plan.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <numbers>
#include <type_traits>
#include <vector>

#include "fft/exceptions.hpp"
#include "fft/fft_plan.hpp"
#include "fft_internal.hpp"

namespace {

/**
 * @brief Complex product with explicit real arithmetic.
 *
 * Avoids the NaN-recovery path of std::complex operator*.
 */
template<typename T>
inline std::complex<T> cmul(std::complex<T> a, std::complex<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}

} // namespace

namespace fft {

/**
 * @brief Internal state of RealFFTPlan.
 *
 * Member order matters: the constructor initializes them in declaration order.
 */
template<typename T>
struct RealFFTPlan<T>::Impl {
    std::size_t n;                  ///< Real sequence length N.
    std::size_t half;               ///< Inner complex length: N/2 (even) or N (odd).
    bool even;                      ///< True when N is even.
    Normalization normalization;    ///< Scaling convention.
    FFTPlan<T> inner;               ///< Unscaled complex plan of length @c half.
    std::vector<std::complex<T>> twiddles; ///< W^k for k = 0..half (even N only).

    Impl(std::size_t size, Normalization norm)
        : n(size),
          half(size % 2 == 0 ? size / 2 : size),
          even(size % 2 == 0),
          normalization(norm),
          inner(half, Normalization::None) {
        if (even) {
            twiddles.resize(half + 1);
            const double base = -2.0 * std::numbers::pi / static_cast<double>(n);
            for (std::size_t k = 0; k <= half; ++k) {
                const double angle = base * static_cast<double>(k);
                twiddles[k] = {static_cast<T>(std::cos(angle)),
                               static_cast<T>(std::sin(angle))};
            }
        }
    }
};

template<typename T>
RealFFTPlan<T>::RealFFTPlan(std::size_t size, Normalization normalization) {
    if (size == 0)
        throw InvalidSizeException("RealFFTPlan length must be greater than zero");
    impl_ = std::make_unique<Impl>(size, normalization);
}

template<typename T>
RealFFTPlan<T>::~RealFFTPlan() = default;

template<typename T>
RealFFTPlan<T>::RealFFTPlan(RealFFTPlan&&) noexcept = default;

template<typename T>
RealFFTPlan<T>& RealFFTPlan<T>::operator=(RealFFTPlan&&) noexcept = default;

template<typename T>
std::size_t RealFFTPlan<T>::size() const noexcept {
    return impl_ ? impl_->n : 0;
}

template<typename T>
std::size_t RealFFTPlan<T>::spectrum_size() const noexcept {
    return impl_ ? impl_->n / 2 + 1 : 0;
}

template<typename T>
Normalization RealFFTPlan<T>::normalization() const noexcept {
    return impl_ ? impl_->normalization : Normalization::None;
}

template<typename T>
Precision RealFFTPlan<T>::precision() const noexcept {
    if constexpr (std::is_same_v<T, float>)
        return Precision::Float32;
    else
        return Precision::Float64;
}

template<typename T>
std::size_t RealFFTPlan<T>::workspace_size() const noexcept {
    if (!impl_)
        return 0;
    // Two buffers of length half, followed by the inner plan's scratch.
    return 2 * impl_->half + impl_->inner.workspace_size();
}

template<typename T>
void RealFFTPlan<T>::execute_forward(const T* in, std::complex<T>* out) const {
    std::vector<std::complex<T>> scratch(workspace_size());
    execute_forward(in, out, scratch.data(), scratch.size());
}

template<typename T>
void RealFFTPlan<T>::execute_forward(const T* in, std::complex<T>* out,
                                     Workspace<T>& workspace) const {
    if (!impl_)
        throw FFTException("RealFFTPlan::execute_forward() called on a moved-from plan");
    const std::size_t need = workspace_size();
    if (workspace.capacity() < need)
        workspace.resize(need);
    execute_forward(in, out, workspace.data(), workspace.capacity());
}

template<typename T>
void RealFFTPlan<T>::execute_forward(const T* in, std::complex<T>* out,
                                     std::complex<T>* scratch,
                                     std::size_t scratch_size) const {
    if (!impl_)
        throw FFTException("RealFFTPlan::execute_forward() called on a moved-from plan");
    const Impl& p = *impl_;

    if (in == nullptr || out == nullptr)
        throw InvalidBufferException("FFT input and output buffers must not be null");
    const std::size_t in_bytes = p.n * sizeof(T);
    const std::size_t out_bytes = (p.n / 2 + 1) * sizeof(std::complex<T>);
    if (detail::ranges_overlap(in, in_bytes, out, out_bytes))
        throw InvalidBufferException("FFT input and output buffers must not overlap");

    const std::size_t need = workspace_size();
    if (scratch == nullptr || scratch_size < need)
        throw InvalidBufferException("FFT scratch buffer is smaller than workspace_size()");
    const std::size_t scratch_bytes = need * sizeof(std::complex<T>);
    if (detail::ranges_overlap(scratch, scratch_bytes, in, in_bytes) ||
        detail::ranges_overlap(scratch, scratch_bytes, out, out_bytes))
        throw InvalidBufferException("FFT scratch buffer must not overlap input or output");

    std::complex<T>* a = scratch;
    std::complex<T>* b = scratch + p.half;
    std::complex<T>* inner_scratch = scratch + 2 * p.half;
    const std::size_t inner_size = p.inner.workspace_size();
    const T scale = static_cast<T>(
        detail::scale_factor(p.normalization, Direction::Forward, p.n));

    if (p.even) {
        const std::size_t m = p.half;
        for (std::size_t j = 0; j < m; ++j)
            a[j] = {in[2 * j], in[2 * j + 1]};
        p.inner.execute(Direction::Forward, a, b, inner_scratch, inner_size);

        // Index k == m wraps to 0 in both lookups, which gives X[m] from Z[0].
        // Compares instead of modulo: an integer division per element dominated the pass.
        for (std::size_t k = 0; k <= m; ++k) {
            const std::size_t kz = (k == m) ? 0 : k;
            const std::size_t kc = (k == 0) ? 0 : m - k;
            const std::complex<T> zk = b[kz];
            const std::complex<T> zc = std::conj(b[kc]);
            const std::complex<T> e = (zk + zc) * T(0.5);
            const std::complex<T> d = (zk - zc) * T(0.5);
            // Division by i: (re + i*im) / i = im - i*re.
            const std::complex<T> o{d.imag(), -d.real()};
            out[k] = (e + cmul(p.twiddles[k], o)) * scale;
        }
    } else {
        for (std::size_t j = 0; j < p.n; ++j)
            a[j] = {in[j], T(0)};
        p.inner.execute(Direction::Forward, a, b, inner_scratch, inner_size);
        for (std::size_t k = 0; k <= p.n / 2; ++k)
            out[k] = b[k] * scale;
    }
}

template<typename T>
void RealFFTPlan<T>::execute_inverse(const std::complex<T>* in, T* out) const {
    std::vector<std::complex<T>> scratch(workspace_size());
    execute_inverse(in, out, scratch.data(), scratch.size());
}

template<typename T>
void RealFFTPlan<T>::execute_inverse(const std::complex<T>* in, T* out,
                                     Workspace<T>& workspace) const {
    if (!impl_)
        throw FFTException("RealFFTPlan::execute_inverse() called on a moved-from plan");
    const std::size_t need = workspace_size();
    if (workspace.capacity() < need)
        workspace.resize(need);
    execute_inverse(in, out, workspace.data(), workspace.capacity());
}

template<typename T>
void RealFFTPlan<T>::execute_inverse(const std::complex<T>* in, T* out,
                                     std::complex<T>* scratch,
                                     std::size_t scratch_size) const {
    if (!impl_)
        throw FFTException("RealFFTPlan::execute_inverse() called on a moved-from plan");
    const Impl& p = *impl_;

    if (in == nullptr || out == nullptr)
        throw InvalidBufferException("FFT input and output buffers must not be null");
    const std::size_t in_bytes = (p.n / 2 + 1) * sizeof(std::complex<T>);
    const std::size_t out_bytes = p.n * sizeof(T);
    if (detail::ranges_overlap(in, in_bytes, out, out_bytes))
        throw InvalidBufferException("FFT input and output buffers must not overlap");

    const std::size_t need = workspace_size();
    if (scratch == nullptr || scratch_size < need)
        throw InvalidBufferException("FFT scratch buffer is smaller than workspace_size()");
    const std::size_t scratch_bytes = need * sizeof(std::complex<T>);
    if (detail::ranges_overlap(scratch, scratch_bytes, in, in_bytes) ||
        detail::ranges_overlap(scratch, scratch_bytes, out, out_bytes))
        throw InvalidBufferException("FFT scratch buffer must not overlap input or output");

    std::complex<T>* a = scratch;
    std::complex<T>* b = scratch + p.half;
    std::complex<T>* inner_scratch = scratch + 2 * p.half;
    const std::size_t inner_size = p.inner.workspace_size();
    const T scale = static_cast<T>(
        detail::scale_factor(p.normalization, Direction::Inverse, p.n));

    // A real signal has real DC and Nyquist bins, so their imaginary parts are dropped.
    const std::complex<T> dc{in[0].real(), T(0)};

    if (p.even) {
        const std::size_t m = p.half;
        const std::complex<T> nyquist{in[m].real(), T(0)};
        auto bin = [&](std::size_t j) -> std::complex<T> {
            if (j == 0) return dc;
            if (j == m) return nyquist;
            return in[j];
        };

        for (std::size_t k = 0; k < m; ++k) {
            const std::complex<T> xk = bin(k);
            const std::complex<T> xc = std::conj(bin(m - k));
            const std::complex<T> e = (xk + xc) * T(0.5);
            // d = W^k * O[k], so O[k] = d * conj(W^k) because |W^k| = 1.
            const std::complex<T> d = (xk - xc) * T(0.5);
            const std::complex<T> o = cmul(d, std::conj(p.twiddles[k]));
            // Z[k] = E[k] + i * O[k].
            const std::complex<T> io{-o.imag(), o.real()};
            a[k] = e + io;
        }
        p.inner.execute(Direction::Inverse, a, b, inner_scratch, inner_size);

        // The inner result is m * (x_even + i * x_odd) and N = 2m, hence the factor 2.
        const T s = T(2) * scale;
        for (std::size_t j = 0; j < m; ++j) {
            out[2 * j] = s * b[j].real();
            out[2 * j + 1] = s * b[j].imag();
        }
    } else {
        // Build the Hermitian extension to the full length N.
        a[0] = dc;
        for (std::size_t k = 1; k <= p.n / 2; ++k) {
            a[k] = in[k];
            a[p.n - k] = std::conj(in[k]);
        }
        p.inner.execute(Direction::Inverse, a, b, inner_scratch, inner_size);
        for (std::size_t j = 0; j < p.n; ++j)
            out[j] = scale * b[j].real();
    }
}

// Explicit instantiation definitions.
template class RealFFTPlan<float>;
template class RealFFTPlan<double>;

} // namespace fft
