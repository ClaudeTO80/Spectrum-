# FFT

A standalone C++20 Fourier transform library. The project is being developed with
1D transforms as the initial implementation target and a core architecture intended
to accommodate future 2D and 3D transforms.

## Current status

**Early development – Phase 2 (core infrastructure).**

The current code provides the initial compiled-library layout, public types,
dimension descriptors, workspace storage, plan metadata, CMake installation and
a basic test/CI setup. The direct DFT (complex, float and double) is implemented; other algorithms are not yet.

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
