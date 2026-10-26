# 0009. Support each optional WebGPU device feature that shaders can use

- Status: Accepted
- Date: 2026-10-23
- Revised: 2026-10-26
- Topic: Targets

## Context

WebGPU has optional device features. Some of them change what a shader can
contain: 16 bit floats, dual source blending, the primitive index, the
subgroup size, more storage texture formats and access modes, and filtering
of 32 bit float textures.

A device that does not have the feature rejects the shader or the pipeline.

The DSL must let users write these shaders. It must also tell the renderer
which features to request. If the DSL uses a feature but does not report it,
the program compiles and then fails on the device.

These results were measured on Chrome 155 (Dawn). Each feature was tested
with and without a request for it.

| Feature | What it adds | WGSL directive |
| --- | --- | --- |
| (core) | Storage textures: rgba8unorm, rgba8snorm, rgba8uint, rgba8sint, rgba16uint, rgba16sint, rgba16float, r32\*, rg32\*, rgba32\*. read_write only for r32float, r32uint and r32sint. | none |
| `bgra8unorm-storage` | bgra8unorm as a storage texture (write and read). | none |
| `texture-formats-tier1` (needs `rg11b10ufloat-renderable`) | Storage for r8\*, r16uint, r16sint, r16float, rg8\*, rg16\*, rgb10a2uint, rgb10a2unorm, rg11b10ufloat, and the new formats r16unorm, r16snorm, rg16unorm, rg16snorm, rgba16unorm, rgba16snorm. | none |
| `texture-formats-tier2` (needs tier1) | read_write for r8unorm, r8uint, r8sint, r16uint, r16sint, r16float, rgba8unorm, rgba8uint, rgba8sint, rgba16uint, rgba16sint, rgba16float, rgba32uint, rgba32sint, rgba32float. | none |
| `shader-f16` | f16 types and `h` literals, also in uniforms. | `enable f16;` |
| `dual-source-blending` | Two outputs at location 0 with `@blend_src(0)` and `@blend_src(1)`. | `enable dual_source_blending;` |
| `primitive-index` | The fragment builtin `primitive_index: u32`. | `enable primitive_index;` |
| `subgroup-size-control` (needs `subgroups`) | `@subgroup_size(N)` on a compute entry point. N is a power of 2. Pipeline creation rejects a workgroup x size that N does not divide. | `enable subgroup_size_control;` |
| `float32-filterable` | Filtered sampling of r32float, rg32float and rgba32float. The renderer selects the format when it binds the texture, so the shader text does not show it. | none |

The other features are about textures, pipelines and queries, not about
shader code: texture compression, `depth-clip-control`,
`depth32float-stencil8`, `timestamp-query`, `indirect-first-instance`,
`rg11b10ufloat-renderable`, `float32-blendable`,
`texture-component-swizzle` and `texture-compression-unaligned`.

`atomic-vec2u-min-max` is a shader feature, but no available adapter has
it. Thus it cannot be tested.

## Decision

### Device features

- The DSL can express each shader feature in the table above.
- Each feature has a `DeviceFeature` value, its WebGPU name
  (`deviceFeatureName()`) and its WGSL extension (`wgslExtension()`), if it
  has one.
- The analysis records each feature that a program uses in
  `Reflection::deviceFeatures`. The list also includes the features that
  they need: tier2 brings tier1 and `rg11b10ufloat-renderable`, and subgroup
  size control brings subgroups. Thus the renderer can give the list
  directly to `requiredFeatures`.
- moga writes WGSL `enable` directives only from that list.
- Device features come from each bound resource, also from the resources
  that `AnalyzeOptions::retain` adds. The reason: WebGPU checks storage
  texture formats when it makes the bind group layout, also when no stage
  uses the texture.

### Language features

- WGSL language features are different from device features. The
  implementation lists them in `navigator.gpu.wgslLanguageFeatures`.
- `Reflection::languageFeatures` lists the language features that a program
  needs. The WGSL code declares them with `requires`:
  - `readonly_and_readwrite_storage_textures`, for read and read_write
    storage textures, and for `textureBarrier()`
  - `texture_formats_tier1`, for the tier 1 storage formats
- Chrome 155 accepts the directive. It reports an unknown name as an error.

### Features that the shader text cannot show

- The user declares these features. For example, a texture that holds 32
  bit floats and is filtered says so in `TextureOptions`.
- The reflection gives each sampler its binding type: filtering, non
  filtering or comparison. Thus the renderer can make a bind group layout
  that agrees with it.

### GLSL

- GLSL ES 3.00 rejects the features that WebGL2 does not have, through the
  backend capabilities (ADR 0005).
- Dual source blending is available in GLSL when the renderer enables
  `WEBGL_blend_func_extended` (`GLSLOptions::blendFuncExtended`).

### Not supported yet

- `atomic-vec2u-min-max` waits until a device that has it can test it.

## Consequences

Positive:

- The DSL can express each shader that a WebGPU device with the correct
  features can run. The reflection tells which features to request.
- There is one table for each concern (formats, features, extensions). Thus
  the rules are in one location. To add a feature, you add rows.

Negative:

- The format tables follow the measured behavior of one implementation. You
  must check them again when the specification changes.
- `float32-filterable` depends on a declaration. The DSL cannot compare it
  with the texture that the renderer binds.

## Open work

- `reflectStages` (`src/Analyze.cpp`) merges language features from the
  vertex stage only. Device features come from the two stages. This has no
  effect now, because the only stage specific language feature
  (`textureBarrier()`) is for compute. A future fragment language feature
  would be missing from the `requires` directives.

## Related

- ADR 0005: backend capabilities.
- ADR 0008: WebGPU only features.
