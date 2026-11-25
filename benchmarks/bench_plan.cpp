/**
 * @file bench_plan.cpp
 * @brief Times the forward transform of one plan length.
 *
 * Usage: fft_bench_plan <length> [repetitions]
 *
 * Reports the fastest of the repetitions, which is the least sensitive to
 * scheduling noise. Compare runs only when they come from the same binary and
 * were taken in alternation, since timings on shared machines vary by tens of
 * percent between sessions.
 */

#include <fft/fft.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <length> [repetitions]\n", argv[0]);
        return 2;
    }
    const std::size_t n = std::strtoul(argv[1], nullptr, 10);
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    if (n == 0 || reps <= 0) {
        std::fprintf(stderr, "length and repetitions must be positive\n");
        return 2;
    }

    std::vector<std::complex<double>> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = {std::sin(0.3 * static_cast<double>(i)), std::cos(0.9 * static_cast<double>(i))};

    fft::FFTPlan<double> plan(n, fft::Normalization::None);
    fft::Workspace<double> workspace(plan.workspace_size());

    double best_ms = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        plan.execute(fft::Direction::Forward, in.data(), out.data(), workspace);
        const auto t1 = std::chrono::steady_clock::now();
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::printf("N=%zu algorithm=%d repetitions=%d best=%.4f ms\n",
                n, static_cast<int>(plan.algorithm()), reps, best_ms);
    return 0;
}
