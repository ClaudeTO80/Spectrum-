# FFT

A standalone C++20 Fourier transform library. The project is being developed with
1D transforms as the initial implementation target and a core architecture intended
to accommodate future 2D and 3D transforms.

## Current status

**Early development – Phase 2 (core infrastructure).**

The current code provides the initial compiled-library layout, public types,
dimension descriptors, workspace storage, plan metadata, CMake installation and
a basic test/CI setup. The direct DFT, radix-2, radix-4 and mixed-radix complex FFTs (float and double) are implemented; Bluestein and Rader are not yet.

## Usage

### Basic API

```cpp
#include <fft/fft.hpp>

std::vector<std::complex<double>> x = /* samples */;
auto X = fft::forward(x);   // forward transform, unscaled
auto y = fft::inverse(X);   // inverse transform, divided by N; y ~= x
```

Both functions accept a `std::vector` or a `std::span` of `std::complex<float>`
or `std::complex<double>`. Each call builds a plan internally.

### Advanced API

Use an explicit plan when you need a normalization other than the default, want to
reuse the twiddle table across many transforms, or need to control output buffers:

```cpp
fft::FFTPlan<float> plan(n, fft::Normalization::Ortho);  // unitary scaling
plan.execute(fft::Direction::Forward, in.data(), out.data());
```

Normalization options: `None`, `Forward`, `Inverse` (default), `Ortho`.
Input and output buffers must not overlap (in-place is not supported yet).
The planner uses radix-4 for powers of four, radix-2 for other powers of two, mixed-radix for other composite lengths, and a direct DFT for primes. Mixed-radix execution allocates a scratch buffer of the transform length.
A plan is immutable and can be shared across threads that use their own buffers.
`fft::Workspace` is provided for caller-owned scratch memory, but no transform
uses it yet.

## Requirements

- CMake 3.25 or newer
- C++20 compiler

## Build

```bash
cmake -S . -B build -DFFT_BUILD_SHARED=ON -DFFT_BUILD_STATIC=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake --install build --prefix install
```

## Consume from another CMake project

```cmake
find_package(FFT CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE FFT::FFTStatic)
```

The exported target names are `FFT::FFTStatic` and `FFT::FFTShared`.

## License

MIT. See [LICENSE](LICENSE).
