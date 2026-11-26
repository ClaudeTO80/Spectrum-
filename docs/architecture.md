# Architecture

## Scope

The current implementation covers complex one-dimensional transforms. The core
introduces rank-aware shape metadata so that later 2D and 3D plans can reuse
dimension, size-validation and memory-layout concepts.

## Layers

- **Public API** (`include/fft/`): plans, types, workspaces, shapes and exceptions,
  plus the basic `forward` and `inverse` functions in `simple.hpp`.
- **Metadata**: `Shape`, `Normalization`, `Precision`, `Direction` and `Algorithm`.
- **Planner** (`src/fft_plan.cpp`, `FFTPlan` constructor): selects the algorithm
  from the length and precomputes twiddles, permutations, per-stage tables, chirp
  and kernel data for the chosen algorithm.
- **Multidimensional transforms** (`src/nd_plan.cpp`, `FFTPlanND`): one unscaled 1D plan per
  axis, applied to strided lines copied through a contiguous buffer. Normalization is applied once
  over the total element count.
- **Real transforms** (`src/real_plan.cpp`, `RealFFTPlan`): packs real input into a
  complex inner plan of length N/2 (even N) or N (odd N), then applies the
  Hermitian post-processing or pre-processing pass.
- **Kernels** (`src/fft_plan.cpp`): direct DFT, radix-2, radix-4, mixed-radix
  (recursive), Bluestein and Rader. Each kernel works on caller-provided buffers.
- **Workspace**: caller-owned scratch memory, reusable across executions.

## Algorithm selection

Selection depends only on the length. See `select_algorithm` in `src/fft_plan.cpp`
and the table in the README. The thresholds were chosen from measurements on a
shared development machine; they are documented next to the code.

## Precision

Kernels compute in the plan's precision `T`, except the direct DFT, which
accumulates in double. Twiddle tables are computed in double and rounded to `T`.

## Thread safety

Plans are immutable after construction, so one plan can be used by several threads.
Each call needs its own buffers. A `Workspace` must not be used by two calls at the
same time, and it is not synchronized.

## Memory

- Radix-2, radix-4 and the direct DFT do not allocate during `execute()`.
- Mixed-radix, Bluestein and Rader need `workspace_size()` elements of scratch.
  The workspace overload reuses a caller-owned workspace; the three-argument
  overload allocates one per call.
- Bluestein and Rader build an inner plan of power-of-two or p−1 length at
  construction time.

## ABI

The library explicitly instantiates `FFTPlan<float>` and `FFTPlan<double>`.
The numeric values of the enumerations are part of the binary interface. Consumers
must use a compatible C++ toolchain and runtime. A stable cross-compiler C ABI is
outside the current scope.
