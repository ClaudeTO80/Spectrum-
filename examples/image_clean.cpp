// Image cleaning: load an image with stb, filter each colour channel in the
// 2D frequency domain, and save the result with stb.
//
// Usage:
//   image_clean <input> <output> <lowpass|highpass|bandpass|bandstop> <cutoff_low> [cutoff_high]
//
// Cutoffs are radial frequencies in cycles per pixel, in [0, sqrt(0.5)] (about 0.707).
// The output format follows the extension of <output>: .png, .jpg/.jpeg or .bmp.
// Alpha channels (2- and 4-channel images) are copied unchanged.

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <fft/fft.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s <input> <output> <lowpass|highpass|bandpass|bandstop> "
                 "<cutoff_low> [cutoff_high]\n"
                 "  cutoffs in cycles per pixel, in [0, sqrt(0.5)]\n",
                 program);
}

Spectrum::FilterType parse_type(const std::string& name) {
    if (name == "lowpass") return Spectrum::FilterType::LowPass;
    if (name == "highpass") return Spectrum::FilterType::HighPass;
    if (name == "bandpass") return Spectrum::FilterType::BandPass;
    if (name == "bandstop") return Spectrum::FilterType::BandStop;
    throw std::invalid_argument("unknown filter type: " + name);
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Writes @p pixels in the format chosen by the extension of @p path.
bool save_image(const std::string& path, int width, int height, int channels,
                const unsigned char* pixels) {
    if (ends_with(path, ".png"))
        return stbi_write_png(path.c_str(), width, height, channels, pixels, width * channels) != 0;
    if (ends_with(path, ".jpg") || ends_with(path, ".jpeg"))
        return stbi_write_jpg(path.c_str(), width, height, channels, pixels, 95) != 0;
    if (ends_with(path, ".bmp"))
        return stbi_write_bmp(path.c_str(), width, height, channels, pixels) != 0;
    throw std::invalid_argument("unsupported output extension: " + path);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        print_usage(argv[0]);
        return 2;
    }

    const std::string input = argv[1];
    const std::string output = argv[2];

    try {
        const Spectrum::FilterType type = parse_type(argv[3]);
        const double cutoff_low = std::stod(argv[4]);
        const double cutoff_high = (argc == 6) ? std::stod(argv[5]) : 0.0;

        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* loaded = stbi_load(input.c_str(), &width, &height, &channels, 0);
        if (loaded == nullptr) {
            std::fprintf(stderr, "cannot load %s: %s\n", input.c_str(), stbi_failure_reason());
            return 1;
        }
        std::vector<unsigned char> pixels(loaded, loaded + static_cast<std::size_t>(width) *
                                                           height * channels);
        stbi_image_free(loaded);

        // Two- and four-channel images carry alpha in the last channel: leave it alone.
        const int color_channels = (channels == 2 || channels == 4) ? channels - 1 : channels;
        const auto rows = static_cast<std::size_t>(height);
        const auto cols = static_cast<std::size_t>(width);
        const auto stride = static_cast<std::size_t>(channels);

        std::vector<double> plane(rows * cols);
        for (int c = 0; c < color_channels; ++c) {
            for (std::size_t i = 0; i < plane.size(); ++i)
                plane[i] = static_cast<double>(pixels[i * stride + static_cast<std::size_t>(c)]);

            const auto filtered =
                Spectrum::filter_real_2d(plane, rows, cols, type, cutoff_low, cutoff_high);

            for (std::size_t i = 0; i < filtered.size(); ++i) {
                const double clamped = std::clamp(std::round(filtered[i]), 0.0, 255.0);
                pixels[i * stride + static_cast<std::size_t>(c)] = static_cast<unsigned char>(clamped);
            }
        }

        if (!save_image(output, width, height, channels, pixels.data())) {
            std::fprintf(stderr, "cannot write %s\n", output.c_str());
            return 1;
        }
        std::printf("%s -> %s (%dx%d, %d channels, %s)\n", input.c_str(), output.c_str(),
                    width, height, channels, argv[3]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
