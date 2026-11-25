// Basic API: forward and inverse transforms in one call each.
#include <fft/fft.hpp>

#include <complex>
#include <cstdio>
#include <vector>

int main() {
    const std::vector<std::complex<double>> samples = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};

    const auto spectrum = fft::forward(samples);
    const auto restored = fft::inverse(spectrum);

    for (std::size_t k = 0; k < spectrum.size(); ++k)
        std::printf("X[%zu] = %g %+gi   x[%zu] = %g\n",
                    k, spectrum[k].real(), spectrum[k].imag(), k, restored[k].real());
    return 0;
}
