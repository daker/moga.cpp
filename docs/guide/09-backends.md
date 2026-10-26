# 9. Backends

moga has two backends:

| Backend | Output | Field |
| --- | --- | --- |
| GLSL ES 3.00 (WebGL2) | A vertex shader and a fragment shader. | `BuildResult::glsl` |
| WGSL (WebGPU) | One module with all entry points. | `BuildResult::wgsl` |

The DSL has WebGPU semantics on the two backends. The GLSL backend adds
code where WebGL2 is different.

## GLSL options

Set `BuildOptions::glsl` to a `GLSLOptions` value. Tell moga which WebGL2
extensions and conventions your renderer uses.

| Option | Default | Meaning |
| --- | --- | --- |
| `flipY` | `true` | Negates clip space Y. Then framebuffer memory, `fragCoord()`, derivatives and read back rows agree with WebGPU. The renderer sets `frontFace` to `CW` and flips the image one time when it presents it. |
| `clipControl` | `false` | `EXT_clip_control` sets a depth range of 0 to 1. When it is false, moga changes the depth from WebGPU (0 to w) to WebGL2 (-w to w). |
| `clipDistanceExtension` | `false` | `WEBGL_clip_cull_distance` is available. Necessary for `ClipDistance()`. |
| `multiDraw` | `false` | `WEBGL_multi_draw` is available. Necessary for `drawIndex()`. |
| `firstVertexConvention` | `false` | The renderer set the first vertex convention with `WEBGL_provoking_vertex`. Necessary for `Interpolation::Flat`. |
| `blendFuncExtended` | `false` | `WEBGL_blend_func_extended` is enabled. Necessary for `blendOutput()`. |

When a program needs an option that is not set, the target check gives a
`MissingFeature` error that names the option.

To compile WGSL only, set `options.glsl = std::nullopt`. To compile GLSL
only, set `options.wgsl = false`.

## WGSL output

| Field | Contents |
| --- | --- |
| `code` | The WGSL module. |
| `vertexEntry`, `fragmentEntry` | `vs_main` and `fs_main`. Empty for a compute program. |
| `computeEntry` | `cs_main`. Empty for a render program. |
| `lineMap` | Generated lines and their C++ statements. |

moga writes the `enable` directives from `Reflection::deviceFeatures`, and
the `requires` directives from `Reflection::languageFeatures`.

## Generated names

moga gives each declaration a name with a prefix. Read the names from the
reflection. Do not write them in your renderer.

| Declaration | GLSL | WGSL |
| --- | --- | --- |
| Attribute `position` | `a_position` | `a_position` |
| Varying `normal` | `v_normal` | `v_normal` |
| Output 0 | `o_0` | |
| Uniform block `camera` | block `UB_camera`, instance `ub_camera` | `ub_camera` |
| Texture `albedo` | `t_albedo` | `t_albedo`, sampler `s_albedo` |
| Table `instances` | `tb_instances` | `tb_instances` |

Names that start with `moga_` are reserved. A declaration with such a name
is an error. Two resources that get the same generated name are also an
error.

Locals get a number after their name, for example `color_7`. Thus two
locals with the same name in C++ do not collide.

## Differences that moga handles

| Topic | WebGPU meaning | What GLSL gets |
| --- | --- | --- |
| Clip space | Y up, depth 0 to w. | Depth remap, and Y flip when `flipY` is set. |
| `mod(a, b)` | `a - b * floor(a / b)`. | The same formula. WGSL gets a helper, because WGSL `%` truncates. |
| `select(c, t, f)` | The two values are evaluated. | A helper function when a value has side effects. |
| `smoothstep` | Defined for reversed edges. | A helper with the WGSL formula. |
| `dot` of integer vectors | Valid. | A helper. |
| `inverse(m)` | Not in WGSL. | WGSL gets a helper. |
| `instanceIndex()` | Includes the first instance. | `gl_InstanceID + moga_BaseInstance`. |
| `Interpolation::Flat` | The first vertex. | Needs the first vertex convention. |
| Bitcasts, vector compares, shift amounts, texture sizes of arrays | WGSL rules. | Lowered to agree. |

The helpers are written only when a program uses them.

## Features that only one backend has

| Feature | WebGPU | WebGL2 |
| --- | --- | --- |
| Compute, storage, atomics, barriers, subgroups | Yes | No |
| Arrays of arrays | Yes | No |
| `Interpolation::Linear` | Yes | No |
| f16 types | With `shader-f16` | No |
| `primitiveIndex()` | With `primitive-index` | No |
| `ClipDistance()` | With `clip-distances` | With `WEBGL_clip_cull_distance` |
| Dual source blending | With `dual-source-blending` | With `WEBGL_blend_func_extended` |
| `localInstanceIndex()` | No | Yes |
| `drawIndex()` | No | With `WEBGL_multi_draw` |

When a program uses a feature that a target does not have, the target check
reports it. See [7. Diagnostics](07-diagnostics.md) and
[ADR 0008](../adr/0008-webgpu-only-features.md).

