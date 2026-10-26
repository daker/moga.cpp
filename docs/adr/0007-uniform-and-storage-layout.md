# 0007. Use one uniform layout, one storage layout and frequency based bind groups

- Status: Accepted
- Date: 2026-10-21
- Revised: 2026-10-26
- Topic: Data layout

## Context

GLSL std140 and the WGSL uniform address space have different rules for
some types: bool, mat2 and arrays of scalars.

A CPU side writer must fill a buffer that is valid on the two backends, with
no code for each backend.

Bindings must be predictable. Then renderers can share layouts between
programs.

## Decision

### Uniform layout

- Uniform structs use a layout that is valid under the two rules.
- moga rejects the types whose rules are different.
- WGSL structs get explicit `@align` and `@size` from that layout.
- `UniformWriter` fills a staging buffer from the layout. Matrices are
  column major. The columns of a mat3 are padded to vec4.

### Single uniforms

- `uniform()` makes a single uniform.
- moga packs the single uniforms of each frequency into one generated block:
  `frameUniforms`, `objectUniforms` or `materialUniforms`.
- The fields are sorted by name.
- The block uses the same layout rules. It shows in the reflection as a
  usual block, so `UniformWriter` can fill it.

### Bindings

- Each resource has an update frequency: frame, object or material.
- The frequencies go to WGSL bind groups 0, 1 and 2.
- In a group, resources are sorted by kind (uniform block, table, texture),
  then by name.
- A sampled texture uses a second binding for its sampler.
- GLSL uniform block points and texture units use the same order.
- `AnalyzeOptions::retain` keeps resources in a program that does not use
  them. Thus programs can share the same frame group.

### Tables

- A table holds a record for each instance. The fields are vec4 or mat4.
- In WGSL, a table becomes a read only storage buffer.
- In GLSL, a table becomes an RGBA32F texture with generated fetch code.

### Storage layout (WebGPU only)

- Storage buffers use the WGSL storage address space layout
  (`storageLayout()`).
- A vec3 aligns to 16 and has a size of 12.
- Arrays and structs are not padded to 16.
- The reflection gives the element layout and the stride.
- One struct cannot be in a uniform block and in a storage buffer, because
  WGSL writes uniform structs with explicit std140 offsets.

## Consequences

Positive:

- The two backends use one buffer upload path.
- Bind group layouts are deterministic. Programs can share them.

Negative:

- bool, mat2 and scalar arrays cannot be in uniform blocks. Use u32,
  mat3 or mat4, and vec4 arrays.
- A struct that is used in uniform and in storage memory must be declared
  two times.

## Related

- ADR 0008: WebGPU only features (storage buffers).
