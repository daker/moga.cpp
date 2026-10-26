#pragma once

// Usage analysis and binding assignment shared by both backends.

#include "moga/Capabilities.hpp"
#include "moga/Layout.hpp"
#include "moga/Nodes.hpp"

#include <array>
#include <memory>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace moga {

/** Returns the bind group of an update frequency: frame 0, object 1, material 2. */
constexpr int groupOfFrequency(ResourceFrequency f) { return static_cast<int>(f); }

/** Number of texels in one row of a GLSL table texture. */
inline constexpr int kTableTextureWidth = 2048;

/**
 * A generated helper function. A backend lowers an expression to one when the
 * lowering would repeat an operand that has side effects.
 */
enum class HelperKind {
  /** WGSL: matrix inverse (always a helper; WGSL has none). */
  Inverse,
  /** WGSL: float modulo a - b * floor(a / b). */
  FloatMod,
  /** GLSL: dot product of integer vectors. */
  IntegerDot,
  /** GLSL: select of an integer or bool vector by a bool vector. */
  VectorSelect,
  /** GLSL: select by a bool that evaluates both values, as WGSL select() does. */
  ScalarSelect,
  /** WGSL: mat3 from the upper left of a mat4. */
  Mat3FromMat4,
  /** GLSL: smoothstep by the WGSL formula, defined for reversed edges. */
  Smoothstep
};

/** One helper function that a backend must emit, for one operand type. */
struct Helper {
  /** The kind of helper. */
  HelperKind kind;
  /** The operand type (the matrix, the vector or the selected type). */
  TypePtr type;
  /** Returns true when the kind and the operand type are the same. */
  bool operator==(const Helper& o) const { return kind == o.kind && type == o.type; }
};

/** A vertex attribute that the vertex stage reads. */
struct ReflectedAttribute {
  /** The name given to attribute(). */
  std::string name;
  TypePtr type;
  /** The shader location of the attribute. */
  int location;
  /** Advances per vertex or per instance. */
  VertexStep step;
};

/** A varying that the fragment stage reads. */
struct ReflectedVarying {
  /** The name given to varying(). */
  std::string name;
  TypePtr type;
  /** The shader location of the varying. */
  int location;
  /** How the varying is interpolated. */
  Interpolation interpolation;
};

/** A fragment output (a color attachment). */
struct ReflectedOutput {
  /** The location of the output (the color attachment index). */
  int index;
  TypePtr type;
  /** The blend source of a dual source blending output. */
  std::optional<int> blendSrc = std::nullopt;
};

/** A uniform block that the program binds. */
struct ReflectedUniformBlock {
  /** The name of the block. */
  std::string name;
  /** The update frequency. It gives the WGSL bind group. */
  ResourceFrequency frequency;
  /** Shares ownership of the block struct; pass it to UniformWriter. */
  std::shared_ptr<const StructLayout> layout;
  /** The WGSL bind group. */
  int group;
  /** The WGSL binding in the group. */
  int binding;
  /** The GLSL interface block name. */
  std::string glslBlockName;
  /** The GLSL uniform buffer binding point. */
  int glslBindingPoint;
};

/** The type of a sampler binding (GPUSamplerBindingLayout.type). */
enum class SamplerBinding {
  /** No sampler is paired. */
  None,
  /** "filtering": float textures. */
  Filtering,
  /** "non-filtering": unfilterable float textures, and depth textures sampled without comparison. */
  NonFiltering,
  /** "comparison": depth comparison. */
  Comparison
};

/** A sampled or loaded texture that the program binds. */
struct ReflectedTexture {
  /** The name given to texture(). */
  std::string name;
  TextureDimension dimension;
  TextureSampleType sampleType;
  ResourceFrequency frequency;
  /** The WGSL bind group. */
  int group;
  /** The WGSL binding of the texture. */
  int binding;
  /** WGSL sampler binding in the same group, -1 when unsampled. */
  int samplerBinding;
  /** The name of the GLSL sampler uniform. */
  std::string glslName;
  /** The GLSL texture unit. */
  int glslUnit;
  /** The sampler binding type to use in the bind group layout. */
  SamplerBinding sampler = SamplerBinding::None;
};

/**
 * A table that the program binds. WGSL reads it as a read only storage
 * buffer. GLSL reads it as an RGBA32F texture.
 */
struct ReflectedTable {
  /** The name given to table(). */
  std::string name;
  ResourceFrequency frequency;
  /** vec4 texels (16 bytes) per record. */
  int stride;
  /** The texel offset of each field in a record. */
  TableFieldOffsets fieldOffsets;
  /** The WGSL bind group. */
  int group;
  /** The WGSL binding. */
  int binding;
  /** The name of the GLSL texture uniform. */
  std::string glslName;
  /** The GLSL texture unit. */
  int glslUnit;
};

/** A storage buffer that the program binds. WebGPU only. */
struct ReflectedStorageBuffer {
  /** The name given to storage(). */
  std::string name;
  ResourceFrequency frequency;
  /** Read or ReadWrite. */
  StorageAccess access;
  /** The type of one element of the runtime sized array. */
  TypePtr elementType;
  /** Layout of one element; `stride` is the distance between elements. */
  StorageLayout element;
  /** The WGSL bind group. */
  int group;
  /** The WGSL binding. */
  int binding;
};

/** A storage texture that the program binds. WebGPU only. */
struct ReflectedStorageTexture {
  /** The name given to storageTexture(). */
  std::string name;
  ResourceFrequency frequency;
  TextureDimension dimension;
  /** The texel format of the storage texture. */
  TexelFormat format;
  StorageAccess access;
  /** The WGSL bind group. */
  int group;
  /** The WGSL binding. */
  int binding;
};

/** The reflection of one bound resource, of any kind. */
using ReflectedResource = std::variant<ReflectedUniformBlock, ReflectedTexture, ReflectedTable,
                                       ReflectedStorageBuffer, ReflectedStorageTexture>;

/** Returns the WGSL bind group of a resource. */
int groupOf(const ReflectedResource& r);
/** Returns the WGSL binding of a resource. */
int bindingOf(const ReflectedResource& r);

/**
 * What the renderer needs to know about a program: its inputs, outputs,
 * bound resources, builtins and required features.
 */
struct Reflection {
  /** The program name. */
  std::string program;
  /** The variant key of the program. */
  std::string variantKey;
  /** The vertex attributes that the vertex stage reads. */
  std::vector<ReflectedAttribute> attributes;
  /** The varyings that the fragment stage reads. */
  std::vector<ReflectedVarying> varyings;
  /** The fragment outputs. */
  std::vector<ReflectedOutput> outputs;
  /** The bound uniform blocks. The generated blocks of single uniforms are included. */
  std::vector<ReflectedUniformBlock> uniformBlocks;
  /** The bound textures. */
  std::vector<ReflectedTexture> textures;
  /** The bound tables. */
  std::vector<ReflectedTable> tables;
  /** The bound storage buffers. WebGPU only. */
  std::vector<ReflectedStorageBuffer> storageBuffers;
  /** The bound storage textures. WebGPU only. */
  std::vector<ReflectedStorageTexture> storageTextures;
  /** Resources of each WGSL bind group, in binding order. */
  std::array<std::vector<ReflectedResource>, 3> groups;
  /** The fragment stage writes FragDepth(). */
  bool usesFragDepth = false;
  /** The number of clip distances that the vertex stage writes: the highest index plus 1. */
  int clipDistances = 0;
  /** The builtins that the vertex stage reads. */
  std::vector<BuiltinInput> vertexBuiltins;
  /** The builtins that the fragment stage reads. */
  std::vector<BuiltinInput> fragmentBuiltins;
  /** The program is a compute program. */
  bool compute = false;
  /** The workgroup size of a compute program (@workgroup_size). */
  std::array<std::uint32_t, 3> workgroupSize{1, 1, 1};
  /** The subgroup size that the compute program asks for, if any. */
  std::optional<std::uint32_t> subgroupSize;
  /** The builtins that the compute stage reads. */
  std::vector<BuiltinInput> computeBuiltins;
  /**
   * The WebGPU device features the program needs, first use first, for every
   * bound resource (retained ones too). Request them, or reject the program.
   */
  std::vector<DeviceFeature> deviceFeatures;
  /** The WGSL language features the program needs (see navigator.gpu.wgslLanguageFeatures). */
  std::vector<LanguageFeature> languageFeatures;
  /** The features that not every backend has, first use first; see unsupported(). */
  std::vector<FeatureUse> features;

  /** Returns true when the varying crosses to the fragment stage (the fragment stage reads it). */
  bool hasVarying(std::string_view name) const;
  /** Returns true when the vertex stage reads this builtin. */
  bool usesVertexBuiltin(BuiltinInput b) const;
  /** Returns true when the fragment stage reads this builtin. */
  bool usesFragmentBuiltin(BuiltinInput b) const;
  /** Returns true when the compute stage reads this builtin. */
  bool usesComputeBuiltin(BuiltinInput b) const;
  /** Returns true when the program needs this device feature. */
  bool needs(DeviceFeature f) const;
};

/** A resource that the program binds, with its reflection. */
struct BoundResource {
  /** The declaration of the resource. */
  ResourcePtr decl;
  /** Where the resource is bound. */
  ReflectedResource reflected;
};

/**
 * A program after analysis. It is checked and has its reflection, and its
 * declarations are in emit order. Both backends compile it.
 */
struct AnalyzedProgram {
  /** The analyzed program; owned, so the result never outlives it. */
  std::shared_ptr<const ProgramAST> ast;
  /** The reflection of the program. */
  Reflection reflection;
  /** User functions. A called function comes before its callers. */
  std::vector<FunctionPtr> functions;
  /** The user functions that the vertex stage calls. A called function comes first. */
  std::vector<FunctionPtr> vertexFunctions;
  /** The user functions that the fragment stage calls. A called function comes first. */
  std::vector<FunctionPtr> fragmentFunctions;
  /** The user functions that the compute stage calls. A called function comes first. */
  std::vector<FunctionPtr> computeFunctions;
  /** Workgroup arrays of a compute program. */
  std::vector<WorkgroupPtr> workgroups;
  /** The struct types that the program declares. A nested struct comes first. */
  std::vector<TypePtr> structs;
  /** Named constants. A constant comes after the constants that its value uses. */
  std::vector<ConstantPtr> constants;
  /** Bound resources in binding order. */
  std::vector<BoundResource> resources;
  /** Each single uniform and the generated block that holds it. */
  std::vector<std::pair<UniformPtr, UniformBlockPtr>> uniforms;
  /** Helper functions that the GLSL backend must emit, first use first. */
  std::vector<Helper> glslHelpers;
  /** Helper functions that the WGSL backend must emit, first use first. */
  std::vector<Helper> wgslHelpers;

  /** Returns the generated block that holds a single uniform, or null. */
  const UniformBlockResource* blockOf(const UniformDecl* uniform) const;

  /** Returns the reflection of a resource, or null when the program does not use it. */
  const ReflectedResource* find(const Resource* decl) const;
};

/** Options of the usage analysis. */
struct AnalyzeOptions {
  /**
   * Resources to bind when the program does not use them.
   * Use this to give all programs the same frame group layout.
   */
  std::vector<ResourcePtr> retain;
};

} // namespace moga
