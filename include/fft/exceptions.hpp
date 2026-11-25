#pragma once

/**
 * @file exceptions.hpp
 * @brief Exception hierarchy of the library.
 *
 * All exceptions derive from FFTException, which derives from std::runtime_error.
 */

#include <stdexcept>
#include <string>

namespace fft {

/** @brief Base class of all exceptions thrown by the library. */
class FFTException : public std::runtime_error {
public:
    /** @param message Human-readable description of the failure. */
    explicit FFTException(const std::string& message) : std::runtime_error(message) {}
};

/** @brief A length or extent is invalid, for example zero. */
class InvalidSizeException : public FFTException {
public:
    /** @param message Human-readable description of the failure. */
    explicit InvalidSizeException(const std::string& message) : FFTException(message) {}
};

/** @brief A buffer pointer is null, or input and output buffers overlap. */
class InvalidBufferException : public FFTException {
public:
    /** @param message Human-readable description of the failure. */
    explicit InvalidBufferException(const std::string& message) : FFTException(message) {}
};

/**
 * @brief Reserved for workspace failures.
 *
 * Not thrown by the current implementation: an undersized workspace is grown
 * instead of rejected.
 */
class WorkspaceException : public FFTException {
public:
    /** @param message Human-readable description of the failure. */
    explicit WorkspaceException(const std::string& message) : FFTException(message) {}
};

/**
 * @brief Reserved for planning failures.
 *
 * Not thrown by the current implementation: every length has an algorithm.
 */
class PlanningException : public FFTException {
public:
    /** @param message Human-readable description of the failure. */
    explicit PlanningException(const std::string& message) : FFTException(message) {}
};

} // namespace fft
