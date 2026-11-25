#pragma once
#include <stdexcept>
#include <string>

namespace fft {

class FFTException : public std::runtime_error {
public:
    explicit FFTException(const std::string& message) : std::runtime_error(message) {}
};

class InvalidSizeException : public FFTException {
public:
    explicit InvalidSizeException(const std::string& message) : FFTException(message) {}
};

class InvalidBufferException : public FFTException {
public:
    explicit InvalidBufferException(const std::string& message) : FFTException(message) {}
};

class WorkspaceException : public FFTException {
public:
    explicit WorkspaceException(const std::string& message) : FFTException(message) {}
};

class PlanningException : public FFTException {
public:
    explicit PlanningException(const std::string& message) : FFTException(message) {}
};

} // namespace fft
