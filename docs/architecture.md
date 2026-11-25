# Architecture

## Scope

The first implementation target is 1D FFT. The core introduces rank-aware shape
metadata so later 2D/3D plans can reuse dimension, size-validation and memory-layout
concepts without forcing multidimensional execution into the 1D release.

## Layers

- Public API: stable user-facing plans, types and exceptions.
- Core metadata: shape, normalization, precision and algorithm identifiers.
- Planner: algorithm selection and plan construction (subsequent phase).
- Execution: algorithm kernels (subsequent phase).
- Workspace: caller-owned scratch memory, reusable across executions.

## Thread-safety direction

Plans should become immutable after construction. Concurrent execution should use
independent workspaces. A workspace must not be used concurrently by multiple calls.

## ABI

The library is compiled and explicitly instantiates float and double plan types.
Consumers must use a compatible C++ toolchain/runtime. A stable cross-compiler C ABI
is outside the initial scope.
