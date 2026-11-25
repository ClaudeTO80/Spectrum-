#include <fft/fft.hpp>
#include <iostream>

int main() {
    fft::FFTPlan<double> plan(1024);
    std::cout << "FFT plan size: " << plan.size() << '\n';
    std::cout << "Normalization: "
              << static_cast<int>(plan.normalization()) << '\n';
    std::cout << "Algorithm selected: "
              << static_cast<int>(plan.algorithm()) << '\n';
}
