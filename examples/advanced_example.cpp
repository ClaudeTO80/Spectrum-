// Advanced API: reuse an explicit plan with a chosen normalization.
#include <fft/fft.hpp>

#include <complex>
#include <cstdio>
#include <vector>

int main() {
    // Ortho normalization makes the transform unitary (energy preserving).
    Spectrum::FFTPlan<float> plan(8, Spectrum::Normalization::Ortho);

    std::vector<std::complex<float>> in(8, {1.0f, 0.0f});
    std::vector<std::complex<float>> out(8);
    plan.execute(Spectrum::Direction::Forward, in.data(), out.data());

    std::printf("size=%zu algorithm=%d X[0]=%g\n",
                plan.size(), static_cast<int>(plan.algorithm()), out[0].real());
    return 0;
}
