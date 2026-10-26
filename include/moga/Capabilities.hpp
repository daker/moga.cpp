#pragma once

// Features that a program uses and the features that each backend supports.
// Analysis records the features; a backend compares them with its set.

#include "moga/Nodes.hpp"

#include <bitset>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moga {

/** A language feature that not every backend has. */
enum class Feature {
  /** A compute program. */
  ComputeProgram,
  /** A storage buffer. */
  StorageBuffer,
  /** A storage texture. */
  StorageTexture,
  /** A workgroup array. */
  WorkgroupArray,
  /** An atomic operation. */
  Atomics,
  /** A workgroup, storage or texture barrier. */
  Barrier,
  /** A compute stage builtin, for example globalId(). */
  ComputeBuiltin,
  /** A subgroup builtin or function. */
  Subgroups,
  /** An array whose elements are arrays. */
  ArrayOfArrays,
  /** A varying with Interpolation::Linear. */
  LinearInterpolation,
  /** The primitiveIndex() builtin. */
  PrimitiveIndex,
  /** A flat varying that takes the value of the first vertex. */
  FlatFirstVertex,
  /** WebGL2 has it with WEBGL_blend_func_extended. */
  DualSourceBlending,
  /** 16 bit floats; GLSL ES 3.00 has none (mediump is a hint, not a type). */
  F16,
  /** The number of features. Not a feature. */
  Count
};

/**
 * An optional WebGPU device feature that the device must have to run the
 * program. It lists every feature that a construct of the DSL can need.
 *
 * One feature that shaders can use is not expressed yet:
 *   atomic-vec2u-min-max    atomicMin and atomicMax on vec2u; no available
 *                           adapter has it, so it cannot be tested
 *
 * The other features (texture compression, depth-clip-control,
 * depth32float-stencil8, timestamp-query, indirect-first-instance,
 * float32-blendable, texture-component-swizzle, texture-compression-unaligned)
 * concern pipelines, textures and queries, not shaders; the renderer requests
 * them. rg11b10ufloat-renderable is listed here only because
 * texture-formats-tier1 requires it. See ADR 0009.
 */
enum class DeviceFeature {
  /** Subgroup builtins and functions ("subgroups"). */
  Subgroups,
  /** A bgra8unorm storage texture ("bgra8unorm-storage"). */
  Bgra8UnormStorage,
  /** Clip distance outputs ("clip-distances"). */
  ClipDistances,
  /** Filtered sampling of 32 bit float textures ("float32-filterable"). */
  Float32Filterable,
  /** The primitiveIndex() fragment builtin ("primitive-index"). */
  PrimitiveIndex,
  /** Required by texture-formats-tier1 ("rg11b10ufloat-renderable"). */
  Rg11b10UfloatRenderable,
  /** More storage texture formats ("texture-formats-tier1"). */
  TextureFormatsTier1,
  /** read_write access for more storage texture formats ("texture-formats-tier2"). */
  TextureFormatsTier2,
  /** A compute program with a fixed subgroup size ("subgroup-size-control"). */
  SubgroupSizeControl,
  /** blendOutput(0) and blendOutput(1) ("dual-source-blending"). */
  DualSourceBlending,
  /** f16 scalars, vectors and matrices ("shader-f16"). */
  ShaderF16
};

/**
 * A WGSL language feature that the program needs. The implementation lists
 * the ones it has in navigator.gpu.wgslLanguageFeatures; the WGSL code
 * declares them with a `requires` directive.
 */
enum class LanguageFeature {
  /** Read and read_write storage textures, and textureBarrier(). */
  ReadonlyAndReadwriteStorageTextures,
  /** The texture-formats-tier1 storage texture formats in WGSL. */
  TextureFormatsTier1
};

/** Returns the WGSL name, for example "readonly_and_readwrite_storage_textures". */
std::string_view languageFeatureName(LanguageFeature f);

/** Returns the name to pass in GPUDeviceDescriptor.requiredFeatures, for example "subgroups". */
std::string_view deviceFeatureName(DeviceFeature f);

/**
 * Returns the WGSL extension that the feature needs, for example
 * "clip_distances" for `enable clip_distances;`, or nothing when the shader
 * text needs no directive (bgra8unorm-storage).
 */
std::optional<std::string_view> wgslExtension(DeviceFeature f);

/** One use of a feature, with the name of what uses it, for example "storage buffer data". */
struct FeatureUse {
  /** The feature that is used. */
  Feature feature;
  /** What uses the feature, for messages. */
  std::string what;

  /** Returns true when the feature and the text are the same. */
  bool operator==(const FeatureUse&) const = default;
};

/** The features that a backend can compile. */
struct BackendCapabilities {
  /** The name in messages, for example "WebGL2". */
  std::string_view name;
  /** One bit for each Feature that the backend can compile. */
  std::bitset<static_cast<std::size_t>(Feature::Count)> supported;

  /** Returns true when the backend can compile this feature. */
  bool supports(Feature f) const { return supported.test(static_cast<std::size_t>(f)); }
};

/** Returns the capabilities of WGSL for WebGPU: all features. */
BackendCapabilities webgpuCapabilities();

/**
 * Returns the capabilities of GLSL ES 3.00 for WebGL2. It has none of the
 * features, with two exceptions. Flat first vertex varyings need
 * `firstVertexConvention` (WEBGL_provoking_vertex). Dual source blending
 * needs `blendFuncExtended` (WEBGL_blend_func_extended).
 */
BackendCapabilities webgl2Capabilities(bool firstVertexConvention, bool blendFuncExtended = false);

/**
 * Returns the device features that a storage texture of this format and
 * access needs, with the features they require (tier2 needs tier1, which
 * needs rg11b10ufloat-renderable), or nothing when WebGPU has no such
 * storage texture (for example a read_write rgba8snorm texture).
 */
std::optional<std::vector<DeviceFeature>> storageFormatFeatures(TexelFormat format, StorageAccess access);

/** Returns the uses that the backend does not support, first use first. */
std::vector<FeatureUse> unsupported(const std::vector<FeatureUse>& uses, const BackendCapabilities& backend);

} // namespace moga
