// End-to-end test of image cleaning: stb I/O around filter_real_2d.
//
// A clean synthetic RGB image is corrupted with deterministic noise and written
// as PNG (lossless). The test checks that:
//  - the PNG round trip through stb is exact;
//  - low-pass filtering each channel improves the PSNR against the clean image.

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
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define IMAGE_TEST_CHECK(cond)                                                    \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                         \
        }                                                                         \
    } while (false)

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr int kChannels = 3;

/// Clean RGB image made of smooth, low-frequency patterns (well below the cutoff).
std::vector<unsigned char> make_clean_image() {
    std::vector<unsigned char> img(static_cast<std::size_t>(kWidth) * kHeight * kChannels);
    const double two_pi = 2.0 * std::numbers::pi;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const double u = static_cast<double>(x) / kWidth;
            const double v = static_cast<double>(y) / kHeight;
            const double r = 128.0 + 90.0 * std::sin(two_pi * 2.0 * u);
            const double g = 128.0 + 90.0 * std::cos(two_pi * 1.0 * v);
            const double b = 128.0 + 60.0 * std::sin(two_pi * (u + v));
            const std::size_t base = (static_cast<std::size_t>(y) * kWidth + x) * kChannels;
            img[base + 0] = static_cast<unsigned char>(std::lround(r));
            img[base + 1] = static_cast<unsigned char>(std::lround(g));
            img[base + 2] = static_cast<unsigned char>(std::lround(b));
        }
    }
    return img;
}

/// Adds deterministic uniform noise in [-amplitude, amplitude], clamped to 0..255.
std::vector<unsigned char> add_noise(const std::vector<unsigned char>& clean, int amplitude,
                                     std::uint32_t seed) {
    std::vector<unsigned char> out(clean.size());
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < clean.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        const double unit = static_cast<double>(state) / 2147483648.0 - 1.0; // [-1, 1)
        const double value = clean[i] + std::lround(unit * amplitude);
        out[i] = static_cast<unsigned char>(std::clamp(value, 0.0, 255.0));
    }
    return out;
}

/// Low-pass filters every channel with a radial mask and rounds back to 8 bits.
std::vector<unsigned char> clean_with_lowpass(const std::vector<unsigned char>& noisy,
                                              double cutoff) {
    const auto rows = static_cast<std::size_t>(kHeight);
    const auto cols = static_cast<std::size_t>(kWidth);
    std::vector<unsigned char> out(noisy.size());
    std::vector<double> plane(rows * cols);
    for (int c = 0; c < kChannels; ++c) {
        for (std::size_t i = 0; i < plane.size(); ++i)
            plane[i] = noisy[i * kChannels + static_cast<std::size_t>(c)];
        const auto filtered = Spectrum::filter_real_2d(plane, rows, cols,
                                                       Spectrum::FilterType::LowPass, cutoff);
        for (std::size_t i = 0; i < filtered.size(); ++i) {
            const double v = std::clamp(std::round(filtered[i]), 0.0, 255.0);
            out[i * kChannels + static_cast<std::size_t>(c)] = static_cast<unsigned char>(v);
        }
    }
    return out;
}

/// PSNR in dB between two 8-bit buffers of equal length.
double psnr(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    double mse = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(a.size());
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

/// Writes a PNG and reads it back; returns the decoded pixels, or an empty vector on error.
std::vector<unsigned char> png_round_trip(const std::filesystem::path& path,
                                          const std::vector<unsigned char>& pixels) {
    const std::string file = path.string();
    if (stbi_write_png(file.c_str(), kWidth, kHeight, kChannels, pixels.data(),
                       kWidth * kChannels) == 0)
        return {};

    int w = 0, h = 0, ch = 0;
    unsigned char* loaded = stbi_load(file.c_str(), &w, &h, &ch, 0);
    if (loaded == nullptr)
        return {};
    std::vector<unsigned char> back(loaded, loaded + static_cast<std::size_t>(w) * h * ch);
    stbi_image_free(loaded);
    if (w != kWidth || h != kHeight || ch != kChannels)
        return {};
    return back;
}

} // namespace

int main() {
    const auto clean = make_clean_image();
    const auto noisy = add_noise(clean, 40, 2024u);

    const auto dir = std::filesystem::temp_directory_path();
    const auto noisy_path = dir / "fft_image_test_noisy.png";
    const auto cleaned_path = dir / "fft_image_test_cleaned.png";

    // Round trip through stb must be bit-exact for PNG.
    const auto noisy_back = png_round_trip(noisy_path, noisy);
    IMAGE_TEST_CHECK(noisy_back == noisy);

    const auto cleaned = clean_with_lowpass(noisy_back, 0.1);
    const auto cleaned_back = png_round_trip(cleaned_path, cleaned);
    IMAGE_TEST_CHECK(cleaned_back == cleaned);

    const double psnr_noisy = psnr(clean, noisy);
    const double psnr_cleaned = psnr(clean, cleaned_back);
    std::printf("PSNR noisy: %.2f dB, cleaned: %.2f dB (gain %.2f dB)\n",
                psnr_noisy, psnr_cleaned, psnr_cleaned - psnr_noisy);
    IMAGE_TEST_CHECK(psnr_cleaned > psnr_noisy + 6.0);

    std::error_code ec;
    std::filesystem::remove(noisy_path, ec);
    std::filesystem::remove(cleaned_path, ec);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("image tests passed\n");
    return 0;
}
