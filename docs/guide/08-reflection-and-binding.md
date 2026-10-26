# 8. Reflection and binding

`BuildResult::program->reflection` tells the renderer how to bind the
program. It is the same for the two backends.

```cpp
const Reflection& r = result.program->reflection;
```

## Vertex inputs

`r.attributes` lists the attributes that the vertex stage reads.

| Field | Use |
| --- | --- |
| `name` | The name given to `attribute()`. |
| `type` | The type. Use it to select the vertex format. |
| `location` | The shader location. |
| `step` | `Vertex` or `Instance`. Use it for the step mode of the buffer. |

The two backends use the same locations. moga sorts the attributes by name
and gives them locations 0, 1, 2 and so on. Always read the location from
the reflection.

## Fragment outputs

`r.outputs` lists the outputs, with `index` (the color attachment) and
`blendSrc` for dual source blending.

## Bind groups

Each resource goes to the WebGPU bind group of its frequency:

| Group | Frequency |
| --- | --- |
| 0 | `Frame` |
| 1 | `Object` |
| 2 | `Material` |

In a group, resources are sorted by kind, then by name. The kind order is:
uniform block, table, texture, storage buffer, storage texture. A sampled
texture takes two bindings: the texture, then its sampler.

`r.groups[g]` lists the resources of group `g` in binding order. Each entry
is a `ReflectedResource`, a `std::variant` of the reflected resource types.
`groupOf()` and `bindingOf()` read the group and the binding of any entry.

```cpp
for (std::size_t g = 0; g < r.groups.size(); ++g)
  for (const ReflectedResource& res : r.groups[g])
    std::println("group {} binding {}", g, bindingOf(res));
```

The same lists are also available by kind: `r.uniformBlocks`,
`r.textures`, `r.tables`, `r.storageBuffers` and `r.storageTextures`.

### Share the frame group between programs

A program binds only the resources that it uses. Thus two programs can have
different frame groups. To give them the same layout, retain the shared
resources:

```cpp
BuildOptions options;
options.analyze.retain = {camera.resource(), lights.resource()};
```

A retained resource is bound also when the program does not use it.

## Bind on WebGPU

1. Make one `GPUBindGroupLayout` for each group from `r.groups`.
2. For a texture, use `ReflectedTexture::sampleType` and `dimension`.
3. For its sampler, use `ReflectedTexture::sampler`:
   `Filtering`, `NonFiltering` or `Comparison`. `samplerBinding` gives
   the binding.
4. Request the features in `r.deviceFeatures` in
   `GPUDeviceDescriptor.requiredFeatures`. Use `deviceFeatureName()` to get
   the WebGPU names.
5. Use `result.wgsl->vertexEntry` and `fragmentEntry` (`vs_main` and
   `fs_main`) as the entry points.

## Bind on WebGL2

GLSL ES 3.00 has no binding qualifier. Thus the renderer must set the
binding points and texture units after it links the program.

For each uniform block:

```js
const index = gl.getUniformBlockIndex(program, block.glslBlockName);
gl.uniformBlockBinding(program, index, block.glslBindingPoint);
gl.bindBufferBase(gl.UNIFORM_BUFFER, block.glslBindingPoint, buffer);
```

For each texture and each table:

```js
gl.useProgram(program);
gl.uniform1i(gl.getUniformLocation(program, texture.glslName), texture.glslUnit);
gl.activeTexture(gl.TEXTURE0 + texture.glslUnit);
gl.bindTexture(target, glTexture);
```

### Per draw uniforms

| Uniform | When | Value |
| --- | --- | --- |
| `CompiledGLSL::baseInstanceUniform` (`moga_BaseInstance`) | The program reads `instanceIndex()`. | The first instance of the draw. WebGL2 does not add it. |
| `CompiledGLSL::drawBaseUniform` (`moga_DrawBase`) | The program reads `drawIndex()`. | The draw base of the multi draw call. |

When the field is empty, the program does not have the uniform.

## Fill uniform blocks

`UniformWriter` fills a CPU buffer in the layout of a block. The same
bytes are valid for WebGL2 and WebGPU.

```cpp
const ReflectedUniformBlock& block = r.uniformBlocks.front();
UniformWriter writer(block.layout);
writer.setMat4("viewProj", viewProj);   // 16 floats, column major
writer.setVec4("eye", 0.0f, 2.0f, 5.0f, 1.0f);
writer.setFloat("time", 1.5f);
writer.setInt("waves", 3);
upload(writer.bytes());
```

| Setter | Field type |
| --- | --- |
| `setFloat` | f32, or f16 (rounded to the nearest f16). |
| `setInt`, `setUint` | i32, u32. |
| `setVec`, `setIVec`, `setUVec` | Vectors. You can give fewer values than components. |
| `setVec4` | vec4f. |
| `setMat3` | mat3x3f. 9 floats, column major. moga adds the padding. |
| `setMat4` | mat4x4f. 16 floats, column major. |
| `setArrayVec4` | One element of a vec4 array. |

- A setter throws when the field does not exist or has a different type.
- `offsetOf(field)` gives the byte offset of a field.
- A writer can use an external buffer:
  `UniformWriter(layout, std::span<std::byte>(...))`.
- The layout of each field is in `block.layout->fields` (name, offset, size
  and alignment).

Single uniforms are in the generated blocks `frameUniforms`,
`objectUniforms` and `materialUniforms`. Fill them in the same way.

## Fill tables

A table has `stride` vec4 texels for each record. `fieldOffsets` gives the
texel offset of each field. A vec4 field is 1 texel. A mat4 field is 4
texels (one for each column).

- **WebGPU**: upload the records to a storage buffer. Each record is
  `stride * 16` bytes.
- **WebGL2**: upload the records to an RGBA32F texture with a width of
  `kTableTextureWidth` (2048) texels. Texel `t` of the table is at
  `(t % 2048, t / 2048)`. Record `i`, field offset `k` is texel
  `i * stride + k`.

## Device and language features

| Field | Use |
| --- | --- |
| `r.deviceFeatures` | The WebGPU device features to request. The features that they need are included. |
| `r.languageFeatures` | The WGSL language features that the code requires. Compare with `navigator.gpu.wgslLanguageFeatures`. |
| `r.features` | Each feature that not all backends have, with the name of what uses it. |

To find out if WebGL2 can run the program before you build GLSL:

```cpp
const std::vector<FeatureUse> missing =
    unsupported(r.features, webgl2Capabilities(/*firstVertexConvention=*/false));
```

## Other reflection data

| Field | Meaning |
| --- | --- |
| `program`, `variantKey` | The name and the variant key. |
| `varyings` | The varyings that the fragment stage reads. |
| `usesFragDepth` | The fragment stage writes `FragDepth()`. |
| `clipDistances` | The number of clip distances: the highest index plus 1. |
| `vertexBuiltins`, `fragmentBuiltins`, `computeBuiltins` | The builtins that each stage reads. |
| `compute`, `workgroupSize`, `subgroupSize` | Compute program data. |
