# 0008. Support WebGPU only features in WGSL and reject them in GLSL

- Status: Accepted
- Date: 2026-10-22
- Revised: 2026-10-26
- Topic: Targets

## Context

WebGPU has features that WebGL2 (GLSL ES 3.00) does not have:

- compute programs
- storage buffers and storage textures
- workgroup memory and barriers
- atomics
- subgroups

One shader source must still compile to the two backends. When only one
backend can run a program, the build must fail early with a clear message.
It must not fail with a driver error, and it must not silently do nothing.

## Decision

### WGSL supports the features

The DSL supports these features, and the WGSL backend writes them:

- `computeProgram()` (through `buildComputeShader()`) with a workgroup size
  and the compute builtins
- `storage()` buffers, with `Read` or `ReadWrite` access
- `storageTexture()` with `store()` and `load()`
- `workgroupArray()`
- the three barriers
- atomics on atomic elements of read/write buffers and of workgroup arrays
- subgroup builtins and functions (`enable subgroups;`)

### GLSL rejects them

- The GLSL backend does not emulate these features.
- The analysis records each WebGPU only feature that a program uses, in
  `Reflection::features`, first use first (ADR 0005).
- The target check step reports each feature that WebGL2 does not support,
  as one diagnostic for each feature (ADR 0004). An example message:
  `k: a compute program needs WebGPU; WebGL2 cannot run it`.
- In the same way, the WGSL target check rejects the GLSL only `drawIndex`
  and `localInstanceIndex`.

### Device features

Some of these features also need an optional WebGPU device feature:
subgroups (`subgroups`), bgra8unorm storage textures (`bgra8unorm-storage`)
and clip distances (`clip-distances`). ADR 0009 gives the rules.

### Stage rules

The DSL and the analysis apply the WebGPU stage rules when the program is
built:

- The vertex stage cannot use read/write storage or a writable storage
  texture.
- Barriers, workgroup arrays and compute builtins are for the compute stage
  only.
- Subgroups are for the compute and fragment stages only.

### Atomics

- A read modify write atomic puts the old value in a new constant. Thus the
  operation runs one time, also when the code does not use the result.
- Only `atomicStore()` writes an atomic. A plain assignment to an atomic is
  an error.

### Storage textures

- A storage texture with `ReadWrite` access in core WebGPU takes only
  r32float, r32uint and r32sint. ADR 0009 gives the formats that device
  features add.

Storage buffer layout is in ADR 0007.

## Consequences

Positive:

- A program that needs WebGPU says so when it is built, with the feature
  name.
- The compiler does not depend on a renderer.

Negative:

- A WebGL2 renderer gets no compute fallback. It needs a different program,
  for example a fragment pass into a render target.
- Before a renderer selects a backend, it must read the diagnostics, or call
  `unsupported(reflection.features, webgl2Capabilities(...))`.

## Related

- ADR 0004: build entry point and diagnostics.
- ADR 0005: backend capabilities.
- ADR 0007: storage layout.
- ADR 0009: optional WebGPU device features.
