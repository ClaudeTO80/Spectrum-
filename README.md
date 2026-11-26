# FFT

A standalone C++20 Fourier transform library. The current release implements
complex one-dimensional transforms in single and double precision, with an
architecture prepared for 2D and 3D transforms.

## Status

**Early development (0.1.0).**

Implemented:

- Complex 1D forward and inverse transforms, `float` and `double`.
- Real-to-complex and complex-to-real transforms, returning the N/2+1 non-redundant bins.
- Algorithms selected by length: radix-4, radix-2, mixed-radix, Bluestein, Rader,
  and a direct DFT for small primes (see [Algorithms](#algorithms)).
- A basic one-call API and an advanced plan-based API.
- Caller-owned workspaces for allocation-free execution.
- Batch execution, sequential or on several threads.
- Multidimensional complex transforms, ranks 1 to 3 (and any rank in the template).

Not yet implemented: explicit SIMD kernels (see the roadmap for the evaluation). See [docs/roadmap.md](docs/roadmap.md).

## Usage

### Basic API

```cpp
#include <fft/fft.hpp>

#include <complex>
#include <vector>

int main() {
    std::vector<std::complex<double>> x = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
    auto X = fft::forward(x);   // forward transform, unscaled
    auto y = fft::inverse(X);   // inverse transform, divided by N; y == x up to rounding
}
```

Both functions accept a `std::vector` or a `std::span` of `std::complex<float>` or
`std::complex<double>`. Each call builds a plan internally, so use the advanced API
when the same length is transformed repeatedly.

### Advanced API

Use an explicit plan to choose a normalization, reuse the plan across calls, or
provide your own scratch memory:

```cpp
#include <fft/fft.hpp>

#include <complex>
#include <vector>

int main() {
    const std::size_t n = 360;
    fft::FFTPlan<float> plan(n, fft::Normalization::Ortho);   // unitary scaling

    std::vector<std::complex<float>> in(n), out(n);
    fft::Workspace<float> ws(plan.workspace_size());          // scratch, reusable

    plan.execute(fft::Direction::Forward, in.data(), out.data(), ws);
}
```

Normalization options:

| Value | Forward | Inverse |
|-------|---------|---------|
| `None` | no scaling | no scaling |
| `Forward` | divide by N | no scaling |
| `Inverse` (default) | no scaling | divide by N |
| `Ortho` | divide by √N | divide by √N |

Input and output buffers must not overlap. In-place execution is not supported yet.

### Batch execution

`execute_batch` runs many transforms of the same length, reading and writing at a
fixed element distance. The sequential overload reuses one workspace. The threaded
overload splits the batch into blocks, one per thread, each with its own workspace:

```cpp
#include <fft/fft.hpp>

#include <complex>
#include <vector>

int main() {
    const std::size_t n = 1024, count = 256;
    std::vector<std::complex<double>> in(n * count, {1.0, 0.0}), out(n * count);
    fft::FFTPlan<double> plan(n);
    fft::Workspace<double> ws;
    plan.execute_batch(fft::Direction::Forward, in.data(), out.data(), count, n, n, ws);
    plan.execute_batch(fft::Direction::Forward, in.data(), out.data(), count, n, n, 0u);  // hardware threads
}
```

Output transforms must not overlap (`out_distance >= size()` when `count > 1`), and the
input and output extents must not overlap. The plan is immutable, so threads share it.
The library links `Threads` on both targets. On 2 cores, two threads gave speedups from
1.2 to 2.1 on batches of 1000 to 4096 points (double, shared VM; the spread is run-to-run noise).

### Multidimensional transforms

`fft::FFTPlanND<T, Rank>` transforms a row-major grid of any rank. Each axis is a
1D transform, and the normalization is applied once over all elements. Explicit
instantiations exist for ranks 1, 2 and 3, and both precisions.

```cpp
#include <fft/fft.hpp>

#include <complex>
#include <vector>

int main() {
    const std::size_t rows = 64, cols = 48;
    std::vector<std::complex<double>> img(rows * cols, {1.0, 0.0});
    fft::FFTPlanND<double, 2> plan(fft::Shape<2>({rows, cols}));
    plan.execute_inplace(fft::Direction::Forward, img.data());  // DC is rows*cols
    plan.execute_inplace(fft::Direction::Inverse, img.data());  // back to ones
}
```

The simple API provides `forward_2d`, `inverse_2d`, `forward_3d` and `inverse_3d`.
Measured (double, shared 2-core VM, five alternating runs): a 2D 1024x1024 transform
takes 16 to 22 ns per element with the blocked line gather (8 adjacent strided lines
per pass), against 24 to 32 ns per element for the one-line-at-a-time gather. For 3D
64x64x64 the blocked gather is about 10% faster by median (19 against 21 ns per element,
thirteen alternating runs in total), with heavy noise on this VM.

### Real transforms

A real sequence has a Hermitian spectrum, so only the `N/2 + 1` bins `X[0] … X[N/2]`
are returned. The basic API provides this directly:

```cpp
#include <fft/fft.hpp>

#include <iostream>
#include <vector>

int main() {
    std::vector<double> signal(1000, 1.0);
    auto spectrum = fft::forward_real(signal);      // 501 complex bins
    auto back = fft::inverse_real(spectrum, 1000);  // equals signal
    std::cout << spectrum.size() << ' ' << back[0] << '\n';
}
```

The advanced API is `fft::RealFFTPlan<T>`, with `execute_forward(const T*, complex<T>*)`
and `execute_inverse(const complex<T>*, T*)`, and the same workspace overloads as
`FFTPlan`. For even N it packs the input into an N/2-point complex transform, which
costs about half of a complex transform of length N. For odd N it runs a full-length
complex transform. The imaginary parts of the DC and Nyquist bins are ignored by
the inverse.

### Workspace

`plan.workspace_size()` reports the scratch elements that `execute()` needs:

- `0` for radix-2, radix-4, and the direct DFT;
- `size()` for mixed-radix;
- twice the convolution length for Bluestein and Rader.

`RealFFTPlan::workspace_size()` is twice the inner length plus the inner plan's scratch.

Pass a `fft::Workspace` to reuse its memory across calls, and across plans, as long
as no two calls use it at the same time. The three-argument `execute()` creates a
temporary workspace, so it allocates for the algorithms above.

### Thread safety

A plan is immutable after construction and can be shared between threads. Each
thread must use its own input and output buffers and its own `Workspace`.

## Algorithms

The planner selects an algorithm from the length alone:

| Length | Algorithm |
|--------|-----------|
| power of four (4, 16, 64, …) | radix-4 |
| other powers of two | radix-2 |
| prime p ≤ 64 | direct DFT |
| prime p > 64, with p−1 a power of two, p < 2³² | Rader |
| other prime p > 64 | Bluestein |
| composite with a prime factor above 64 | Bluestein |
| other composites | mixed-radix |

`plan.algorithm()` returns the selected algorithm. Real transforms of even length use
the table for N/2; odd lengths use it for N.

## Requirements

- CMake 3.25 or newer
- A C++20 compiler (GCC, Clang or MSVC)

## Build

```bash
cmake -S . -B build -DFFT_BUILD_SHARED=ON -DFFT_BUILD_STATIC=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake --install build --prefix install
```

To build the timing tool used for performance work:

```bash
cmake -S . -B build -DFFT_BUILD_BENCHMARKS=ON
cmake --build build --config Release
./build/fft_bench_plan 4096 200
```

Timings on shared machines vary by tens of percent between sessions. Compare
binaries only when they are run in alternation on the same machine.

## Consume from another CMake project

```cmake
find_package(FFT CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE FFT::FFTStatic)
```

The exported target names are `FFT::FFTStatic` and `FFT::FFTShared`.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)

## License

MIT. See [LICENSE](LICENSE).
