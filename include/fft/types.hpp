#pragma once
#include <cstddef>
#include <cstdint>

namespace fft {

enum class Direction : std::uint8_t { Forward, Inverse };
enum class Normalization : std::uint8_t { None, Forward, Inverse, Ortho };
enum class Algorithm : std::uint8_t {
    Unselected, DirectDFT, Radix2, Radix4, MixedRadix, Bluestein, Rader
};
enum class Precision : std::uint8_t { Float32, Float64 };

} // namespace fft
