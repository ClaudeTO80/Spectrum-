#include <fft/fft.hpp>

#include <cassert>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>

int main() {
    fft::FFTPlan<float> pf(256);
    assert(pf.size() == 256);
    assert(pf.precision() == fft::Precision::Float32);
    assert(pf.normalization() == fft::Normalization::Inverse);
    assert(pf.shape().element_count() == 256);

    fft::FFTPlan<double> pd(1024, fft::Normalization::Ortho);
    assert(pd.size() == 1024);
    assert(pd.precision() == fft::Precision::Float64);
    assert(pd.normalization() == fft::Normalization::Ortho);

    fft::Shape<2> shape2({64, 32});
    assert(shape2.element_count() == 2048);

    bool caught = false;
    try {
        fft::FFTPlan<double> invalid(0);
    } catch (const fft::InvalidSizeException&) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        fft::Shape<2> invalid_shape({4, 0});
    } catch (const fft::InvalidSizeException&) {
        caught = true;
    }
    assert(caught);

    fft::Workspace<double> workspace(128);
    assert(workspace.capacity() == 128);

    std::cout << "FFT core tests passed\n";
    return 0;
}
