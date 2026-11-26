# Roadmap

Status as of 0.1.0.

1. Core and build infrastructure: **done**
2. Direct DFT, radix-2 and radix-4 complex FFT: **done**
3. Mixed-radix: **done** (recursive, table-driven)
4. Bluestein: **done** (chirp-z, power-of-two convolution)
5. Rader: **done** for primes with p-1 a power of two; composite p-1 is measured
   slower than Bluestein and is not selected
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
   - Pending: SIMD kernels (not started; a portable version needs per-ISA code
     paths and an MSVC build, so it is a separate decision), data movement between
     mixed-radix passes, which is not yet measured separately, so Rader can cover
     composite p-1, and per-stage tables for the direct DFT
8. Batch execution and parallelism: pending
9. 2D and 3D extensions: pending
