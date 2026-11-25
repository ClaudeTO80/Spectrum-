#include "fft/fft_plan.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#include "fft/exceptions.hpp"

namespace fft {

namespace {

/// Pairs (i, r) with i < r whose elements are exchanged to reach a reordered layout.
using SwapList = std::vector<std::pair<std::size_t, std::size_t>>;

/**
 * @brief Returns true if @p n is a power of four greater than or equal to four.
 *
 * Lengths of one (4^0) are handled by the radix-2 path, which is a plain copy.
 */
bool is_power_of_four(std::size_t n) {
    return n >= 4 && std::has_single_bit(n) && (std::countr_zero(n) % 2 == 0);
}

/**
 * @brief Returns the prime factors of @p n in non-decreasing order, with multiplicity.
 * @param n Transform length, greater than zero.
 */
std::vector<std::size_t> prime_factors(std::size_t n) {
    std::vector<std::size_t> factors;
    for (std::size_t p = 2; p * p <= n; ++p) {
        while (n % p == 0) {
            factors.push_back(p);
            n /= p;
        }
    }
    if (n > 1)
        factors.push_back(n);
    return factors;
}

/// Lengths and prime factors above this use Bluestein instead of direct or mixed-radix steps.
constexpr std::size_t kBluesteinThreshold = 64;

/**
 * @brief Length of the power-of-two convolution used by Bluestein for length n.
 * @param n Transform length, greater than one.
 */
std::size_t bluestein_convolution_length(std::size_t n) {
    return std::bit_ceil(2 * n - 1);
}

/**
 * @brief Selects the algorithm for a transform length.
 *
 * Radix4 for powers of four, Radix2 for other powers of two, DirectDFT for
 * primes, MixedRadix for other composite lengths.
 */
Algorithm select_algorithm(std::size_t n) {
    if (is_power_of_four(n))
        return Algorithm::Radix4;
    if (std::has_single_bit(n))
        return Algorithm::Radix2;
    if (n < 2)
        return Algorithm::DirectDFT;  // n == 0 is rejected by the constructor.

    // Large prime factors make the mixed-radix cost grow with the factor, so
    // they are handled by Bluestein; small primes are cheaper done directly.
    const auto factors = prime_factors(n);
    if (factors.size() == 1)
        return n > kBluesteinThreshold ? Algorithm::Bluestein : Algorithm::DirectDFT;
    return factors.back() > kBluesteinThreshold ? Algorithm::Bluestein
                                                : Algorithm::MixedRadix;
}

/**
 * @brief Builds the bit-reversal permutation of a power-of-two length as swaps.
 * @param n Transform length, a power of two.
 */
SwapList build_bit_reversal_swaps(std::size_t n) {
    SwapList swaps;
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            swaps.emplace_back(i, j);
    }
    return swaps;
}

/**
 * @brief Builds the base-4 digit-reversal permutation of a power-of-four length as swaps.
 * @param n Transform length, a power of four.
 */
SwapList build_digit_reversal_swaps(std::size_t n) {
    const std::size_t digits = static_cast<std::size_t>(std::countr_zero(n)) / 2;
    SwapList swaps;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t r = 0;
        std::size_t x = i;
        for (std::size_t d = 0; d < digits; ++d) {
            r = (r << 2) | (x & 3u);
            x >>= 2;
        }
        if (i < r)
            swaps.emplace_back(i, r);
    }
    return swaps;
}

/**
 * @brief Returns the scale factor applied to the result of a transform.
 * @param normalization Plan normalization convention.
 * @param direction     Transform direction.
 * @param n             Transform length.
 */
double scale_factor(Normalization normalization, Direction direction, std::size_t n) {
    const double dn = static_cast<double>(n);
    const bool inverse = direction == Direction::Inverse;
    switch (normalization) {
        case Normalization::None:    return 1.0;
        case Normalization::Forward: return inverse ? 1.0 : 1.0 / dn;
        case Normalization::Inverse: return inverse ? 1.0 / dn : 1.0;
        case Normalization::Ortho:   return 1.0 / std::sqrt(dn);
    }
    return 1.0;
}

/**
 * @brief Returns true if the byte ranges [a, a+bytes) and [b, b+bytes) intersect.
 */
bool ranges_overlap(const void* a, const void* b, std::size_t bytes) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + bytes && pb < pa + bytes;
}

/**
 * @brief Complex product with explicit real arithmetic.
 *
 * std::complex operator* carries NaN recovery logic that may call __muldc3 or
 * __mulsc3. This form avoids that path in the butterfly loops.
 */
template<typename T>
inline std::complex<T> cmul(std::complex<T> a, std::complex<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}

/**
 * @brief Direct O(N^2) DFT, computed in double precision.
 * @param twiddles Table of exp(-2*pi*i*k/N), k in [0, N).
 * @param inverse  True for the inverse transform (conjugated twiddles).
 * @param in       Input samples, N elements, not aliased with @p out.
 * @param out      Output spectrum, N elements, before scaling.
 */
template<typename T>
void direct_dft(const std::vector<std::complex<double>>& twiddles, bool inverse,
                const std::complex<T>* in, std::complex<T>* out) {
    const std::size_t n = twiddles.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            const std::complex<double> w = twiddles[(k * j) % n];
            acc += std::complex<double>(in[j]) * (inverse ? std::conj(w) : w);
        }
        out[k] = std::complex<T>(static_cast<T>(acc.real()), static_cast<T>(acc.imag()));
    }
}

/**
 * @brief Recursive decimation-in-time mixed-radix transform of length n.
 *
 * Splits the length-n transform on the prime factor p = factors[level] into p
 * subsequences of length m = n/p, transforms each recursively into a
 * contiguous block of @p out, then combines them with the identity
 *   X[k + m*q] = sum_r W_n^(r*k) * Y_r[k] * W_p^(r*q).
 *
 * @param twiddles  Table of exp(-2*pi*i*j/N), j in [0, N), N the plan length.
 * @param factors   Prime factors of N, with multiplicity.
 * @param level     Index in @p factors of the factor to split on.
 * @param in        Input; sample j is at in[j * in_stride].
 * @param in_stride Distance between consecutive input samples.
 * @param n         Length of this sub-transform, the product of factors[level..].
 * @param out       Output, n contiguous elements.
 * @param tmp       Scratch, at least n elements; reused by the children, which
 *                  finish before this level uses it.
 * @param inverse   True for the inverse transform (conjugated twiddles).
 */
template<typename T>
void mixed_radix_rec(const std::vector<std::complex<double>>& twiddles,
                     const std::vector<std::size_t>& factors, std::size_t level,
                     const std::complex<T>* in, std::size_t in_stride, std::size_t n,
                     std::complex<T>* out, std::complex<T>* tmp, bool inverse) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }

    const std::size_t N = twiddles.size();
    const std::size_t p = factors[level];
    const std::size_t m = n / p;

    // W_N^j for j < N, conjugated for the inverse transform.
    auto w = [&](std::size_t j) {
        const std::complex<double> wd = twiddles[j];
        return std::complex<T>(static_cast<T>(wd.real()),
                               static_cast<T>(inverse ? -wd.imag() : wd.imag()));
    };

    // Y_r: transform of the subsequence in[r + p*j], stored in out[r*m, (r+1)*m).
    for (std::size_t r = 0; r < p; ++r)
        mixed_radix_rec(twiddles, factors, level + 1, in + r * in_stride, in_stride * p,
                        m, out + r * m, tmp, inverse);

    // Pass 1: apply the cross-stage twiddles W_n^(r*k) into scratch.
    const std::size_t step = N / n;
    for (std::size_t r = 0; r < p; ++r)
        for (std::size_t k = 0; k < m; ++k)
            tmp[r * m + k] = cmul(out[r * m + k], w(r * k * step));

    // Pass 2: length-p DFT across r for each k; outputs go to their final places.
    const std::size_t pstep = N / p;
    for (std::size_t k = 0; k < m; ++k) {
        for (std::size_t q = 0; q < p; ++q) {
            std::complex<T> acc{T(0), T(0)};
            for (std::size_t r = 0; r < p; ++r)
                acc += cmul(tmp[r * m + k], w(((r * q) % p) * pstep));
            out[k + m * q] = acc;
        }
    }
}

/**
 * @brief Bluestein (chirp-z) transform of length n = chirp.size().
 *
 * Uses nk = (n^2 + k^2 - (k-n)^2) / 2 to write the DFT as a convolution:
 *   X_k = conj_chirp_k * sum_j (x_j * conj_chirp_j) * chirp_(k-j),
 * with chirp_j = exp(i*pi*j^2/n) and the sign chosen for the forward DFT.
 * The convolution is computed with length-m power-of-two FFTs, where @p kernel
 * holds the forward transform of the zero-padded, wrapped chirp.
 *
 * The inverse transform uses IDFT(x) = conj(DFT(conj(x))), unscaled.
 *
 * @param chirp   exp(-i*pi*j^2/n) for j in [0, n).
 * @param kernel  Forward transform of the wrapped exp(+i*pi*j^2/n), length m.
 * @param inner   Plan of length m, unnormalized, power of two.
 * @param in      Input, n elements, not aliased with @p out.
 * @param out     Output, n elements, before scaling.
 * @param scratch At least 2*m elements; not aliased with @p in or @p out.
 * @param inverse True for the inverse transform.
 */
template<typename T>
void bluestein_transform(const std::vector<std::complex<T>>& chirp,
                         const std::vector<std::complex<T>>& kernel,
                         const FFTPlan<T>& inner,
                         const std::complex<T>* in, std::complex<T>* out,
                         std::complex<T>* scratch, bool inverse) {
    const std::size_t n = chirp.size();
    const std::size_t m = kernel.size();
    std::complex<T>* a = scratch;
    std::complex<T>* c = scratch + m;

    const T sign = inverse ? T(-1) : T(1);
    for (std::size_t j = 0; j < n; ++j) {
        const std::complex<T> x(in[j].real(), sign * in[j].imag());
        a[j] = cmul(x, chirp[j]);
    }
    std::fill(a + n, a + m, std::complex<T>(T(0), T(0)));

    inner.execute(Direction::Forward, a, c);
    for (std::size_t k = 0; k < m; ++k)
        c[k] = cmul(c[k], kernel[k]);

    // Inverse FFT of c through conjugation: ifft(z) = conj(F(conj(z))) / m.
    for (std::size_t k = 0; k < m; ++k)
        c[k] = std::complex<T>(c[k].real(), -c[k].imag());
    inner.execute(Direction::Forward, c, a);

    const T inv_m = T(1) / static_cast<T>(m);
    for (std::size_t k = 0; k < n; ++k) {
        const std::complex<T> conv(a[k].real() * inv_m, -a[k].imag() * inv_m);
        const std::complex<T> x = cmul(chirp[k], conv);
        out[k] = inverse ? std::complex<T>(x.real(), -x.imag()) : x;
    }
}

/**
 * @brief In-place iterative radix-2 decimation-in-time FFT (Cooley-Tukey).
 *
 * The stage twiddles are read from @p stage_tw, laid out as, for each stage
 * of length len = 2, 4, ..., N, the half = len/2 values W_len^k with
 * W_len = exp(-2*pi*i/len). Access is contiguous in k, and the block loop is
 * the outer loop, as in radix4_in_place.
 *
 * @param stage_tw Stage twiddle table in precision T, built by the plan.
 * @param n        Transform length, a power of two.
 * @param swaps    Bit-reversal permutation, applied before the butterflies.
 * @param inverse  True for the inverse transform (conjugated twiddles).
 * @param data     N elements, transformed in place, before scaling.
 */
template<typename T>
void radix2_in_place(const std::vector<std::complex<T>>& stage_tw, std::size_t n,
                     const SwapList& swaps, bool inverse, std::complex<T>* data) {
    for (const auto& [a, b] : swaps)
        std::swap(data[a], data[b]);

    // Conjugation for the inverse transform, folded into a sign on the imaginary part.
    const T sign = inverse ? T(-1) : T(1);

    std::size_t offset = 0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const std::complex<T>* w = stage_tw.data() + offset;
        offset += half;

        for (std::size_t start = 0; start < n; start += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const std::complex<T> wk(w[k].real(), sign * w[k].imag());
                const std::complex<T> u = data[start + k];
                const std::complex<T> v = cmul(data[start + k + half], wk);
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
    }
}

/**
 * @brief In-place iterative radix-4 decimation-in-time FFT.
 *
 * Each stage combines four sub-transforms of length q = len/4 using the
 * identity X[k + m*q] = sum_r W_len^(r*k) * Y_r[k] * W_4^(r*m), where
 * W_4 = -i for the forward transform and +i for the inverse.
 *
 * The stage twiddles are read from @p stage_tw, laid out as, for each stage,
 * the q values W^k, then the q values W^(2k), then the q values W^(3k), where
 * W = exp(-2*pi*i/len). Access is contiguous in k, and the block loop is the
 * outer loop, so every stage streams through memory once.
 *
 * @param stage_tw Stage twiddle table in precision T, built by the plan.
 * @param n        Transform length, a power of four.
 * @param swaps    Base-4 digit-reversal permutation, applied before the butterflies.
 * @param inverse  True for the inverse transform (conjugated twiddles).
 * @param data     N elements, transformed in place, before scaling.
 */
template<typename T>
void radix4_in_place(const std::vector<std::complex<T>>& stage_tw, std::size_t n,
                     const SwapList& swaps, bool inverse, std::complex<T>* data) {
    for (const auto& [a, b] : swaps)
        std::swap(data[a], data[b]);

    // Multiplication by -i (forward) or +i (inverse) is a swap and sign change.
    auto rotate = [inverse](std::complex<T> d) {
        return inverse ? std::complex<T>(-d.imag(), d.real())
                       : std::complex<T>(d.imag(), -d.real());
    };
    // Conjugation for the inverse transform, folded into a sign on the imaginary part.
    const T sign = inverse ? T(-1) : T(1);

    std::size_t offset = 0;
    for (std::size_t len = 4; len <= n; len <<= 2) {
        const std::size_t q = len >> 2;
        const std::complex<T>* w1 = stage_tw.data() + offset;
        const std::complex<T>* w2 = w1 + q;
        const std::complex<T>* w3 = w2 + q;
        offset += 3 * q;

        for (std::size_t start = 0; start < n; start += len) {
            for (std::size_t k = 0; k < q; ++k) {
                const std::complex<T> c1(w1[k].real(), sign * w1[k].imag());
                const std::complex<T> c2(w2[k].real(), sign * w2[k].imag());
                const std::complex<T> c3(w3[k].real(), sign * w3[k].imag());

                const std::complex<T> a0 = data[start + k];
                const std::complex<T> a1 = cmul(data[start + k + q], c1);
                const std::complex<T> a2 = cmul(data[start + k + 2 * q], c2);
                const std::complex<T> a3 = cmul(data[start + k + 3 * q], c3);

                const std::complex<T> t0 = a0 + a2;
                const std::complex<T> t1 = a0 - a2;
                const std::complex<T> t2 = a1 + a3;
                const std::complex<T> t3 = rotate(a1 - a3);

                data[start + k]         = t0 + t2;
                data[start + k + q]     = t1 + t3;
                data[start + k + 2 * q] = t0 - t2;
                data[start + k + 3 * q] = t1 - t3;
            }
        }
    }
}

} // namespace

/**
 * @brief Private state of an FFTPlan. Null in the owning object once moved from.
 *
 * Holds the precomputed twiddle factors exp(-2*pi*i*k/N) for k in [0, N),
 * computed in double precision regardless of T; the permutation of the
 * selected algorithm as a list of swaps, so that execute() does no index
 * arithmetic for reordering; and, for Radix2 and Radix4, the per-stage twiddle
 * table in precision T, laid out for contiguous access.
 */
template<typename T>
struct FFTPlan<T>::Impl {
    /**
     * @brief Validates the length, precomputes twiddles and the permutation.
     * @throws InvalidSizeException if @p n is zero.
     */
    Impl(std::size_t n, Normalization norm, Algorithm alg)
        : length(n), normalization(norm), algorithm(alg) {
        if (length == 0)
            throw InvalidSizeException("FFT length must be greater than zero");

        twiddles.reserve(length);
        const double step = 2.0 * std::numbers::pi / static_cast<double>(length);
        for (std::size_t k = 0; k < length; ++k)
            twiddles.push_back(std::polar(1.0, -step * static_cast<double>(k)));

        switch (algorithm) {
            case Algorithm::Radix4:
                swaps = build_digit_reversal_swaps(length);
                for (std::size_t len = 4; len <= length; len <<= 2) {
                    const std::size_t q = len >> 2;
                    const std::size_t stride = length / len;
                    for (std::size_t power = 1; power <= 3; ++power)
                        for (std::size_t k = 0; k < q; ++k) {
                            const std::complex<double> wd = twiddles[(power * k * stride) % length];
                            stage_tw.emplace_back(static_cast<T>(wd.real()), static_cast<T>(wd.imag()));
                        }
                }
                break;
            case Algorithm::Radix2:
                swaps = build_bit_reversal_swaps(length);
                for (std::size_t len = 2; len <= length; len <<= 1) {
                    const std::size_t half = len >> 1;
                    const std::size_t stride = length / len;
                    for (std::size_t k = 0; k < half; ++k) {
                        const std::complex<double> wd = twiddles[k * stride];
                        stage_tw.emplace_back(static_cast<T>(wd.real()), static_cast<T>(wd.imag()));
                    }
                }
                break;
            case Algorithm::MixedRadix: factors = prime_factors(length); break;
            case Algorithm::Bluestein: {
                const std::size_t m = bluestein_convolution_length(length);
                const std::uint64_t period = 2 * static_cast<std::uint64_t>(length);
                // exp(+/- i*pi*j^2/n), with j^2 reduced modulo 2n to keep the angle small.
                auto chirp_at = [&](std::size_t j, double sgn) {
                    const std::uint64_t r = (static_cast<std::uint64_t>(j) * j) % period;
                    const double angle = sgn * std::numbers::pi * static_cast<double>(r)
                                       / static_cast<double>(length);
                    return std::polar(1.0, angle);
                };
                chirp.reserve(length);
                for (std::size_t j = 0; j < length; ++j) {
                    const std::complex<double> c = chirp_at(j, -1.0);
                    chirp.emplace_back(static_cast<T>(c.real()), static_cast<T>(c.imag()));
                }
                std::vector<std::complex<T>> wrapped(m, std::complex<T>(T(0), T(0)));
                wrapped[0] = std::complex<T>(T(1), T(0));
                for (std::size_t j = 1; j < length; ++j) {
                    const std::complex<double> c = chirp_at(j, +1.0);
                    const std::complex<T> v(static_cast<T>(c.real()), static_cast<T>(c.imag()));
                    wrapped[j] = v;
                    wrapped[m - j] = v;
                }
                inner = std::make_unique<FFTPlan<T>>(m, Normalization::None);
                kernel.assign(m, std::complex<T>(T(0), T(0)));
                inner->execute(Direction::Forward, wrapped.data(), kernel.data());
                break;
            }
            default: break;
        }
    }

    std::size_t length;
    Normalization normalization;
    Algorithm algorithm;
    std::vector<std::complex<double>> twiddles;
    SwapList swaps;
    std::vector<std::complex<T>> stage_tw;  // Radix2 and Radix4, see the kernels.
    std::vector<std::size_t> factors;       // MixedRadix only: prime factors of length.
    std::vector<std::complex<T>> chirp;     // Bluestein only: exp(-i*pi*j^2/n).
    std::vector<std::complex<T>> kernel;    // Bluestein only: forward transform of the wrapped chirp.
    std::unique_ptr<FFTPlan<T>> inner;      // Bluestein only: power-of-two plan of length kernel.size().
};

template<typename T>
FFTPlan<T>::FFTPlan(std::size_t size, Normalization normalization)
    : impl_(std::make_unique<Impl>(size, normalization, select_algorithm(size))) {}

template<typename T>
FFTPlan<T>::~FFTPlan() = default;

template<typename T>
FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;

template<typename T>
FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan&& other) noexcept = default;

template<typename T>
std::size_t FFTPlan<T>::size() const noexcept {
    return impl_ ? impl_->length : 0;
}

template<typename T>
Shape<1> FFTPlan<T>::shape() const {
    if (!impl_)
        throw FFTException("FFTPlan::shape() called on a moved-from plan");
    return Shape<1>({impl_->length});
}

template<typename T>
Normalization FFTPlan<T>::normalization() const noexcept {
    return impl_ ? impl_->normalization : Normalization::None;
}

template<typename T>
Precision FFTPlan<T>::precision() const noexcept {
    if constexpr (std::is_same_v<T, float>)
        return Precision::Float32;
    else
        return Precision::Float64;
}

template<typename T>
Algorithm FFTPlan<T>::algorithm() const noexcept {
    return impl_ ? impl_->algorithm : Algorithm::Unselected;
}

template<typename T>
std::size_t FFTPlan<T>::workspace_size() const noexcept {
    if (!impl_)
        return 0;
    switch (impl_->algorithm) {
        case Algorithm::MixedRadix: return impl_->length;
        case Algorithm::Bluestein:  return 2 * impl_->kernel.size();
        default:                    return 0;
    }
}

template<typename T>
void FFTPlan<T>::execute(Direction direction,
                         const std::complex<T>* in,
                         std::complex<T>* out) const {
    // Empty until needed: radix paths never touch it, and an empty vector
    // does not allocate.
    Workspace<T> workspace;
    execute(direction, in, out, workspace);
}

template<typename T>
void FFTPlan<T>::execute(Direction direction,
                         const std::complex<T>* in,
                         std::complex<T>* out,
                         Workspace<T>& workspace) const {
    if (!impl_)
        throw FFTException("FFTPlan::execute() called on a moved-from plan");
    if (in == nullptr || out == nullptr)
        throw InvalidBufferException("FFT input and output buffers must not be null");

    const std::size_t n = impl_->length;
    if (ranges_overlap(in, out, n * sizeof(std::complex<T>)))
        throw InvalidBufferException("FFT input and output buffers must not overlap");

    const bool inverse = direction == Direction::Inverse;

    switch (impl_->algorithm) {
        case Algorithm::Radix4:
            std::copy(in, in + n, out);
            radix4_in_place(impl_->stage_tw, n, impl_->swaps, inverse, out);
            break;
        case Algorithm::Radix2:
            std::copy(in, in + n, out);
            radix2_in_place(impl_->stage_tw, n, impl_->swaps, inverse, out);
            break;
        case Algorithm::MixedRadix:
            if (workspace.capacity() < n)
                workspace.resize(n);
            mixed_radix_rec(impl_->twiddles, impl_->factors, 0, in, 1, n, out,
                            workspace.data(), inverse);
            break;
        case Algorithm::Bluestein: {
            const std::size_t need = 2 * impl_->kernel.size();
            if (workspace.capacity() < need)
                workspace.resize(need);
            bluestein_transform(impl_->chirp, impl_->kernel, *impl_->inner,
                                in, out, workspace.data(), inverse);
            break;
        }
        default:
            direct_dft(impl_->twiddles, inverse, in, out);
            break;
    }

    const double scale = scale_factor(impl_->normalization, direction, n);
    if (scale != 1.0) {
        const auto s = static_cast<T>(scale);
        for (std::size_t k = 0; k < n; ++k)
            out[k] *= s;
    }
}

// Explicit instantiation definitions. Export comes from the class definition.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace fft
