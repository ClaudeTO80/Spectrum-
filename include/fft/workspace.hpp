#pragma once

/**
 * @file workspace.hpp
 * @brief Caller-owned scratch memory for plan execution.
 */

#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "exceptions.hpp"

namespace fft {

/**
 * @brief Reusable scratch buffer of complex elements.
 *
 * A workspace is passed to FFTPlan::execute() to avoid allocations on each call.
 * It can be reused across calls and across plans, but a workspace must not be
 * used by two calls at the same time. It is not synchronized.
 *
 * @tparam T Floating-point precision, @c float or @c double.
 *
 * @note Calling resize() may reallocate, which invalidates pointers from data().
 */
template<typename T>
class Workspace {
    static_assert(std::is_floating_point_v<T>,
                  "Workspace precision must be a floating-point type");

public:
    /**
     * @brief Creates a workspace with the given number of complex elements.
     * @param complex_elements Initial capacity; zero allocates nothing.
     */
    explicit Workspace(std::size_t complex_elements = 0)
        : storage_(complex_elements) {}

    /** @brief Number of complex elements the workspace holds. */
    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }

    /**
     * @brief Changes the capacity, preserving no contents.
     * @param complex_elements New capacity in complex elements.
     */
    void resize(std::size_t complex_elements) { storage_.resize(complex_elements); }

    /** @brief Pointer to the first element; valid until the next resize(). */
    [[nodiscard]] std::complex<T>* data() noexcept { return storage_.data(); }

    /** @brief Const pointer to the first element; valid until the next resize(). */
    [[nodiscard]] const std::complex<T>* data() const noexcept { return storage_.data(); }

private:
    std::vector<std::complex<T>> storage_;
};

} // namespace fft
