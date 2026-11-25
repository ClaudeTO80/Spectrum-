# FFT

A standalone C++20 Fourier transform library. The current release implements
complex one-dimensional transforms in single and double precision, with an
architecture prepared for 2D and 3D transforms.

## Status

**Early development (0.1.0).**

Implemented:

- Complex 1D forward and inverse transforms, `float` and `double`.
- Algorithms selected by length: radix-4, radix-2, mixed-radix, Bluestein, Rader,
  and a direct DFT for small primes (see [Algorithms](#algorithms)).
- A basic one-call API and an advanced plan-based API.
- Caller-owned workspaces for allocation-free execution.

Not yet implemented: real-to-complex transforms, SIMD kernels, batch execution and
parallelism, and 2D/3D transforms. See [docs/roadmap.md](docs/roadmap.md).

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

### Workspace

`plan.workspace_size()` reports the scratch elements that `execute()` needs:

- `0` for radix-2, radix-4, and the direct DFT;
- `size()` for mixed-radix;
- twice the convolution length for Bluestein and Rader.

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

`plan.algorithm()` returns the selected algorithm.

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
