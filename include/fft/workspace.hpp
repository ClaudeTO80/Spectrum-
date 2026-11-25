#pragma once

#include <complex>
#include <cstddef>
#include <vector>
#include <type_traits>

#include "exceptions.hpp"

namespace fft {

template<typename T>
class Workspace {
    static_assert(std::is_floating_point_v<T>,
                  "Workspace precision must be a floating-point type");

public:
    explicit Workspace(std::size_t complex_elements = 0)
        : storage_(complex_elements) {}

    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }

    void resize(std::size_t complex_elements) { storage_.resize(complex_elements); }

    [[nodiscard]] std::complex<T>* data() noexcept { return storage_.data(); }
    [[nodiscard]] const std::complex<T>* data() const noexcept { return storage_.data(); }

private:
    std::vector<std::complex<T>> storage_;
};

} // namespace fft
