# 4. Stage inputs, outputs and resources

## Vertex attributes

```cpp
const Val position = attribute("position", vec3f());
const Val offset = attribute("offset", vec3f(), VertexStep::Instance);
```

- The type is a numeric scalar or vector.
- `VertexStep::Vertex` (the default) advances for each vertex.
  `VertexStep::Instance` advances for each instance.
- Read an attribute only in the vertex stage.
- moga assigns the locations. Read them from the reflection. See
  [8. Reflection and binding](08-reflection-and-binding.md).

## Varyings

A varying is a value that the vertex stage writes and the fragment stage
reads.

```cpp
const Varying vNormal = varying("normal", vec3f());
// vertex stage
vNormal.set(normal);
// fragment stage
const Val n = Let(normalize(vNormal.value()), "n");
```

The interpolation mode is the third argument:

| `Interpolation` | Meaning | Backends |
| --- | --- | --- |
| `Perspective` | Perspective correct. The default. | All. |
| `Linear` | Linear in screen space. | WebGPU only. |
| `Flat` | The value of the first vertex. | All. WebGL2 needs the first vertex convention. |
| `FlatEither` | The value of any one vertex. | All. No extension. |

- Integer varyings must use `Flat` or `FlatEither`.
- Use `FlatEither` for values that are the same on each vertex of a
  primitive, for example an instance id.
- For `Flat` on WebGL2, the renderer must call
  `provokingVertexWEBGL(FIRST_VERTEX_CONVENTION_WEBGL)` from
  `WEBGL_provoking_vertex`. Then set `GLSLOptions::firstVertexConvention`.

## Fragment outputs

```cpp
const Output fragColor = output(0);             // vec4f at location 0
const Output normals = output(1, vec4f());
// fragment stage
fragColor.set(vec4(color, 1.0));
```

- The location is 0 to 7.
- Write an output only in the fragment stage.

### Dual source blending

```cpp
const Output color = blendOutput(0);
const Output factor = blendOutput(1);
```

- The two outputs use location 0. They must have the same type.
- No other output can exist.
- WebGPU needs the `dual-source-blending` device feature.
- WebGL2 needs `WEBGL_blend_func_extended` and
  `GLSLOptions::blendFuncExtended`.

## Builtin inputs and outputs

### Inputs

| Function | Type | Stage | Notes |
| --- | --- | --- | --- |
| `vertexIndex()` | u32 | vertex | |
| `instanceIndex()` | u32 | vertex | Includes the first instance of the draw. |
| `localInstanceIndex()` | u32 | vertex | GLSL only. Without the first instance. |
| `drawIndex()` | u32 | vertex | GLSL only. Needs `WEBGL_multi_draw` and `GLSLOptions::multiDraw`. |
| `fragCoord()` | vec4f | fragment | The origin is top left. Pixel centers are at .5. |
| `frontFacing()` | bool | fragment | |
| `primitiveIndex()` | u32 | fragment | WebGPU only. Needs `primitive-index`. |

For the compute builtins, see [10. WebGPU only features](10-webgpu-only.md).

### Outputs

| Function | Stage | Notes |
| --- | --- | --- |
| `Position(v)` | vertex | Clip space position, vec4f. Y points up. Depth is 0 to w. |
| `FragDepth(d)` | fragment | f32. |
| `ClipDistance(i, d)` | vertex | `i` is 0 to 7. A negative value clips. WebGPU needs `clip-distances`. WebGL2 needs `WEBGL_clip_cull_distance` and `GLSLOptions::clipDistanceExtension`. |

Each path through the vertex stage must write `Position()`. moga reports an
error when a path does not.

## Resource frequency

Each resource has a `ResourceFrequency`. It tells how often the renderer
changes the resource. It also selects the WebGPU bind group.

| Frequency | Bind group | Example |
| --- | --- | --- |
| `Frame` | 0 | Camera, time, lights. |
| `Object` | 1 | Model matrix, instance data. |
| `Material` | 2 | Colors, textures. The default for most resources. |

## Single uniforms

```cpp
const Val time = uniform("time", f32(), ResourceFrequency::Frame);
```

moga packs the single uniforms of each frequency into one generated block:
`frameUniforms`, `objectUniforms` or `materialUniforms`. The fields are
sorted by name. Fill the block as any other block.

## Uniform blocks

```cpp
const UniformBlock camera = uniformBlock(
    "camera", {{"viewProj", mat4f()}, {"eye", vec4f()}}, ResourceFrequency::Frame);
// in a stage
Position(camera["viewProj"] * vec4(position, 1.0));
```

- Read a field with `block["field"]`. An unknown field is an error.
- The layout is valid for GLSL std140 and for WGSL. Thus one buffer works
  on the two backends.
- Do not use `bool`, `mat2` or arrays of scalars in a block. Use `u32`,
  `mat3` or `mat4`, and `vec4` arrays.

## Textures

```cpp
const Texture albedo = texture("albedo", TextureDimension::D2);
const Texture shadow = texture("shadow", TextureDimension::D2,
                               {.sampleType = TextureSampleType::Depth, .comparison = true});
```

### Dimensions

| `TextureDimension` | Coordinate | Notes |
| --- | --- | --- |
| `D2` | vec2f | |
| `D3` | vec3f | No depth textures. |
| `D2Array` | vec2f, plus a layer | |
| `Cube` | vec3f direction | No `load()` and no `sampleOffset()`. |

### Options (`TextureOptions`)

| Field | Default | Meaning |
| --- | --- | --- |
| `sampleType` | `Float` | `Float`, `UnfilterableFloat`, `Uint`, `Sint` or `Depth`. `Uint` and `Sint` cannot be sampled. |
| `frequency` | `Material` | The bind group. |
| `sampled` | true for float | Pairs a sampler with the texture. |
| `comparison` | false | Pairs a comparison sampler. Use with a depth texture and `compare()`. |
| `float32Filterable` | false | The texture holds 32 bit floats and is filtered. Needs `float32-filterable`. |

### Operations

| Operation | Use |
| --- | --- |
| `sample(coord, level = 0, layer)` | Filtered sample at an explicit level of detail. |
| `sampleGrad(coord, ddx, ddy, layer)` | Filtered sample with explicit gradients. |
| `sampleOffset(coord, offset, level = 0, layer)` | Sample with a constant texel offset in -8..7. |
| `compare(coord, reference)` | Depth comparison. The result is 0..1. |
| `load(coord, level = 0, layer)` | One texel with integer coordinates. No filter. |
| `size(level = 0)` | The size in texels. |

`sample()` always uses an explicit level. Thus it is valid in non uniform
control flow on WGSL.

## Tables

A table holds records that the shader reads by a u32 index. A common use is
one record for each instance.

```cpp
const Table instances = table(
    "instances", {{"model", mat4f()}, {"tint", vec4f()}}, ResourceFrequency::Object);
// vertex stage
const TableRecord record = instances.at(instanceIndex());
Position(camera["viewProj"] * record["model"] * vec4(position, 1.0));
```

- Fields are `vec4f` or `mat4f` only.
- WGSL reads the table as a read only storage buffer.
- GLSL reads the table from an RGBA32F texture. A vec4 is 1 texel. A mat4
  is 4 texels. A row of the texture has 2048 texels (`kTableTextureWidth`).
- The reflection gives the stride and the field offsets in texels.
