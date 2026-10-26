# 0002. Write shaders in a C++ DSL that compiles to GLSL ES 3.00 and WGSL

- Status: Accepted
- Date: 2026-10-21
- Revised: 2026-10-26
- Topic: Foundation

## Context

Renderers for WebGL2 and WebGPU keep two copies of each shader. The two
copies become different over time.

Each backend needs its own reflection data to bind attributes, uniforms and
textures.

String templates do not find type errors. The driver finds them only when it
compiles the result.

## Decision

- The user builds a shader with C++ code. The code records an abstract
  syntax tree (AST).
- The AST holds attributes, varyings, outputs, uniform blocks, textures,
  tables, locals, control flow and user functions.
- The DSL checks the types of each operation when the shader is built. It
  checks operand shapes, matrix products, swizzles and stage rules.
- The DSL reports each misuse as a diagnostic (ADR 0004).
- An analysis pass finds what each stage uses. It checks the stage interface
  and makes one reflection for the two backends.
- Two backends write GLSL ES 3.00 and WGSL.
- The DSL has WebGPU semantics on the two backends. The backends lower the
  differences: clip space Y and depth, float `mod`, bitcasts, vector
  compares, shift amounts and the texture size of arrays.
- A program variant has a key. The key is made from the sorted options, so
  it is always the same for the same options. Both backends use it as a
  cache key.

The DSL uses the WebGPU meaning where the backends are different:

- `select` evaluates the two values. When a value has side effects, GLSL
  calls a helper function.
- `Interpolation::Flat` uses the first vertex. WebGL2 must set the first
  vertex convention with `WEBGL_provoking_vertex`. The renderer tells this
  to moga with `GLSLOptions::firstVertexConvention`.
- `Interpolation::FlatEither` uses any vertex. It needs nothing more.
- `Interpolation::Linear` has no GLSL ES 3.00 equivalent. It is WebGPU only.

## Consequences

Positive:

- Each shader has one source. The two outputs come from the same AST.
- Type errors show when the shader is built, with a clear message.

Negative:

- The DSL can only express features that the two backends support, or that
  ADR 0008 marks as WebGPU only.
- WGSL rejects the GLSL only builtins `drawIndex` and `localInstanceIndex`.

## Related

- ADR 0003: DSL representation.
- ADR 0004: build entry point and diagnostics.
- ADR 0008: WebGPU only features.
