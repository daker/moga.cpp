# 10. WebGPU only features

The features in this chapter are in WGSL only. The GLSL backend rejects a
program that uses them. They are declared in `moga/dsl/Compute.hpp`.

## Compute programs

```cpp
const StorageBuffer particles = storage("particles", particle);
const Val dt = uniform("dt", f32(), ResourceFrequency::Frame);

const BuildResult r = buildComputeShader("integrate", "INTEGRATE", {64, 1, 1}, [&] {
  const Val i = Let(globalId().x(), "i");
  If(i >= particles.length(), [] { Return(); });
  const Val p = particles[i];
  p.member("position").addAssign(p.member("velocity") * dt);
});
```

- `ComputeOptions::workgroupSize` is the workgroup size. Each size must be
  more than 0.
- `ComputeOptions::subgroupSize` sets a fixed subgroup size. It must be a
  power of 2 that divides `workgroupSize[0]`. The device needs
  `subgroup-size-control`.
- The result has WGSL only. The entry point is `cs_main`.
- `ShaderBuild::finishCompute()` does the same as `buildComputeShader()`
  after a `ShaderBuild`.

### Compute builtins

| Function | Type | Meaning |
| --- | --- | --- |
| `globalId()` | vec3u | The global invocation id. |
| `localId()` | vec3u | The invocation id in its workgroup. |
| `localIndex()` | u32 | The linear invocation index in its workgroup. |
| `workgroupId()` | vec3u | The workgroup id. |
| `numWorkgroups()` | vec3u | The number of workgroups in the dispatch. |

## Storage buffers

```cpp
const StorageBuffer bins = storage("bins", atomicU32());
const StorageBuffer sums = storage("sums", f32(), StorageAccess::ReadWrite);
const StorageBuffer input = storage("input", vec4f(), StorageAccess::Read);
```

- A storage buffer is a runtime sized array of `elementType`.
- `buffer[i]` reads an element. With `ReadWrite` access, you can also write
  it.
- `buffer.length()` gives the number of elements (u32).
- Access is `Read` or `ReadWrite`. `Write` is an error.
- The vertex stage cannot use a `ReadWrite` buffer.
- The default frequency is `Object`.

### Storage layout

Storage buffers use the WGSL storage layout, not the uniform layout:

- A vec3 aligns to 16 bytes and has a size of 12 bytes.
- Arrays and structs are not padded to 16 bytes.

`ReflectedStorageBuffer::element` gives the layout of one element.
`element.stride` is the distance between two elements. `storageLayout(type)`
calculates the layout of a type.

One struct cannot be in a uniform block and in a storage buffer. Declare
two structs.

## Storage textures

```cpp
const StorageTexture speedMap =
    storageTexture("speedMap", TextureDimension::D2, TexelFormat::Rgba8Unorm);
// in a compute stage
speedMap.store(uvec2(x, y), vec4(color, 1.0));
```

- The dimension is `D2`, `D3` or `D2Array`. `Cube` is an error.
- `store(coord, value, layer)` writes one texel. Not valid with `Read`
  access or in the vertex stage.
- `load(coord, layer)` reads one texel. Not valid with `Write` access.
- `size()` gives the size in texels.
- The default access is `Write`.
- Some formats and accesses need a device feature. `storageFormatFeatures()`
  gives them. See "Device features" below.

## Workgroup memory and barriers

```cpp
const Val local = workgroupArray("localBins", atomicU32(), 64);
// in a compute stage
atomicStore(local[localIndex()], 0u);
workgroupBarrier();
```

- `workgroupArray(name, type, length)` declares an array that the
  invocations of one workgroup share. Compute stage only.
- `workgroupBarrier()` makes workgroup memory writes visible.
- `storageBarrier()` makes storage buffer writes visible.
- `textureBarrier()` makes storage texture writes visible.
- All invocations must reach a barrier together. See
  [5. Control flow and functions](05-control-flow-and-functions.md).

## Atomics

The target is an atomic element of a `ReadWrite` storage buffer or of a
workgroup array. Use `atomicU32()` or `atomicI32()` as the element type.

| Function | Effect |
| --- | --- |
| `atomicLoad(t)` | Reads the value. |
| `atomicStore(t, v)` | Writes the value. |
| `atomicAdd`, `atomicSub` | Adds or subtracts. |
| `atomicMin`, `atomicMax` | Keeps the smaller or the larger value. |
| `atomicAnd`, `atomicOr`, `atomicXor` | Bitwise operations. |
| `atomicExchange(t, v)` | Writes `v`. |

- Each read modify write function returns the old value. The operation
  runs one time, also when you do not use the result.
- Only `atomicStore()` writes an atomic. `set()` on an atomic is an error.
- Atomics are not valid in the vertex stage.

## Subgroups

Subgroup operations need the `subgroups` device feature. They are valid in
the compute and fragment stages.

| Function | Result |
| --- | --- |
| `subgroupSize()` | The subgroup size (u32). |
| `subgroupInvocationId()` | The invocation id in the subgroup (u32). |
| `subgroupAdd`, `subgroupMul`, `subgroupMin`, `subgroupMax` | A reduction of a numeric value over the active invocations. |
| `subgroupAnd`, `subgroupOr`, `subgroupXor` | A bitwise reduction of an integer. |
| `subgroupAll(b)`, `subgroupAny(b)` | True when the bool is true in all or in one active invocation. |
| `subgroupBroadcastFirst(v)` | The value of the first active invocation. |
| `subgroupBroadcast(v, id)` | The value of invocation `id`. `id` is a constant u32 in 0..127. |
| `subgroupShuffle(v, id)` | The value of invocation `id`. `id` can be different for each invocation. |
| `subgroupElect()` | True in one active invocation only. |
| `subgroupBallot(b)` | A vec4u bit mask of the invocations where `b` is true. |

## Device features

Some features need an optional WebGPU device feature.
`Reflection::deviceFeatures` lists the features that a program needs, with
the features that they need. Give this list to `requiredFeatures`.

| DSL feature | Device feature |
| --- | --- |
| Subgroup builtins and functions | `subgroups` |
| `ComputeOptions::subgroupSize` | `subgroup-size-control` (and `subgroups`) |
| f16 types and constructors | `shader-f16` |
| `blendOutput()` | `dual-source-blending` |
| `primitiveIndex()` | `primitive-index` |
| `ClipDistance()` | `clip-distances` |
| `TextureOptions::float32Filterable` | `float32-filterable` |
| A `Bgra8Unorm` storage texture | `bgra8unorm-storage` |
| Storage textures in the tier 1 formats | `texture-formats-tier1` (and `rg11b10ufloat-renderable`) |
| `ReadWrite` access in the tier 2 formats | `texture-formats-tier2` (and tier 1) |

In core WebGPU, a `ReadWrite` storage texture can use only `R32Float`,
`R32Uint` and `R32Sint`.

`float32-filterable` depends on the texture that the renderer binds. The
shader text does not show it. Thus you must declare it in
`TextureOptions`. Otherwise, declare such textures with
`TextureSampleType::UnfilterableFloat`.

For the full list and the measured behavior, see
[ADR 0009](../adr/0009-optional-webgpu-device-features.md).

## Example

`examples/10_compute.cpp` has three kernels that share one particle
buffer: an integration step, a histogram with workgroup atomics and a
reduction with subgroups.
