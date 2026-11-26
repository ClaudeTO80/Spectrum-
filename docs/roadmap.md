# Roadmap

Status as of 0.1.0.

1. Core and build infrastructure: **done**
2. Direct DFT, radix-2 and radix-4 complex FFT: **done**
3. Mixed-radix: **done** (recursive, table-driven)
4. Bluestein: **done** (chirp-z, power-of-two convolution)
5. Rader: **done** for primes with p-1 a power of two. Composite p-1 was measured
   again after the symmetric odd-factor butterfly (double, forward, shared VM):
   Rader is correct (error about 1e-13) but 1.2 to 1.6 times slower than Bluestein
   for p from 67 to 4001, so the selection rule is unchanged
6. Real-to-complex and complex-to-real FFT: **done** (`RealFFTPlan`, packed
   N/2-point transform for even N, full-length complex transform for odd N)
7. SIMD and performance work: **in progress**.
   - Done: the first radix-2 stages (lengths 2 and 4) and the first radix-4 stage
     (length 4) no longer multiply by twiddles that equal 1 or ±i. Measured on
     double precision, shared VM: radix-2 lengths 5–16% faster (N=128: 0.73 to
     0.61 µs; N=512: 3.74 to 3.31 µs); radix-4 changes are within the ±5% noise.
   - Measured and not adopted: making the inverse direction a compile-time
     parameter of the radix kernels. Forward improved 5-7% at N=2048 and 8192, the
     inverse did not change, and the difference was within run-to-run noise.
   - Measured: cost per N*log2(N) rises about 55% from N=1024 (0.63 ns) to N=2^18
     and above (0.98 ns), in double precision, consistent with cache misses once
     data exceeds the L2 cache.
   - Measured and not adopted: a four-step prototype (N = R*C, column blocks of 8,
     tiled transpose, twiddles from a table). It is correct (relative error about
     5e-16), but 1.2-1.4 times slower than FFTPlan for N from 2^16 to 2^20. Removing
     the twiddle multiply saves only about 10%, so the cost is in the gathers and
     transposes. An estimate of the ceiling: at N=2^20 the gap to the in-cache rate
     is about 7 ms of the 20 ms total, so blocking could recover at most part of it.
   - Done: closed-form radix-3 and radix-5 butterflies in the mixed-radix path.
     Measured, double precision, forward, shared VM: 243 from 43 to 17 ns per
     element; 125 from 36 to 11; 625 from 46 to 13; 3125 from 57 to 16. Power-of-two
     lengths are unchanged. The generic length-p loop remains for other factors.
   - Done: symmetric butterfly for every odd factor (pairs a_j +/- a_(p-j)), about
     p(p-1) real products instead of 4p^2. Measured, double precision, forward,
     shared VM: 343 from 46 to 20 ns per element; 121 from 46 to 18; 2401 from 60 to
     27; 2197 from 79 to 27; 4004 from 75 to 34. Power-of-two lengths are unchanged.
     A first version with per-call tables was slower (zeroing local arrays per
     recursion node dominated); removing them was necessary.
   - Measured, mixed-radix cost split (forward, double, shared VM), by removing
     one part at a time in a scratch copy: the butterflies take 50-75% of the time
     (for N=1001, about 24 of 32 ns per element); data movement and recursion take
     20-35% (6-14 ns); the twiddle pass takes under 15% (about 5 ns at N=12000).
     Fusing the twiddle pass into the butterfly pass would remove one write and
     one read per level, an estimated gain of about 10%; not implemented.
   - Evaluated SIMD (not implemented). Host: AVX2 and AVX-512. (1) Compiler
     vectorization: -O3 -march=native gains 5-25% over -O2 on the current kernels.
     (2) Split-array layout (re[], im[]) with the same radix-2 algorithm and
     contiguous stage twiddles, auto-vectorized: 1.2-1.4x slower than the current
     interleaved kernel at N=2048 and 8192. The likely cause is the first stages,
     whose inner loops are 1 to 4 elements long and do not vectorize (hypothesis,
     not profiled). A hand-written kernel would need x86 intrinsics, a NEON path,
     a scalar fallback and MSVC testing, for a bounded gain. Not pursued for now.
   - Pending: explicit SIMD kernels, only if vectorized across independent blocks
     is designed (the first stages need it), with a fallback and an MSVC build
     paths and an MSVC build, so it is a separate decision), faster butterflies for
     large odd factors (the main remaining cost), and per-stage tables for the
     direct DFT
8. Batch execution and parallelism: **done** for a fixed length: `execute_batch` (sequential,
   reusing a workspace) and the threaded overload (blocks per thread, one workspace each). The
   library uses `std::thread` and stays C++20. Measured 1.2-2.1x with 2 threads on this VM.
   Tried TK4CPP::ThreadPool (C++23, from ClaudeTO80/TK4CPP, read-only) as an
   optional executor, one batch block per task: on large batches it matches
   std::thread (10.5 vs 10.1 ms for 1024 x 2048; 23.9 vs 24.1 ms for 4096 x 1024).
   On repeated small batches it is about twice as fast (0.016 vs 0.035 ms for 64 x 8),
   because the pool keeps its threads alive. The comparison is not like for like:
   std::thread creates threads on each call. Not added to the library, since the core
   stays C++20.
9. 2D and 3D extensions: pending
