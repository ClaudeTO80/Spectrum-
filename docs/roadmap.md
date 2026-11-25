# Roadmap

Status as of 0.1.0.

1. Core and build infrastructure: **done**
2. Direct DFT, radix-2 and radix-4 complex FFT: **done**
3. Mixed-radix: **done** (recursive, table-driven)
4. Bluestein: **done** (chirp-z, power-of-two convolution)
5. Rader: **done** for primes with p-1 a power of two; composite p-1 is measured
   slower than Bluestein and is not selected
6. Real-to-complex and complex-to-real FFT: pending
7. SIMD and performance work: pending. Candidates: block layouts for the radix-2
   kernel, faster mixed-radix passes so Rader can cover composite p-1, and
   per-stage tables for the direct DFT
8. Batch execution and parallelism: pending
9. 2D and 3D extensions: pending
