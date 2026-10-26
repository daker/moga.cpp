#pragma once

// Stage inputs and outputs, uniform blocks, textures and tables.

#include "moga/dsl/Value.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace moga {

/**
 * Declares a single uniform. Analysis packs the uniforms of each frequency
 * into one generated uniform block, `frameUniforms`, `objectUniforms` or
 * `materialUniforms`, with the fields sorted by name. Fill it as any block.
 */
Val uniform(std::string name, TypePtr type,
            ResourceFrequency frequency = ResourceFrequency::Material, Loc loc = Loc::current());

/** Declares a vertex attribute of a numeric scalar or vector type. Read it only in the vertex stage. */
Val attribute(std::string name, TypePtr type, VertexStep step = VertexStep::Vertex, Loc loc = Loc::current());

/** A value that the vertex stage writes and the fragment stage reads. */
class Varying {
public:
  /** Declares the varying. Integer varyings must use Flat or FlatEither interpolation. */
  Varying(std::string name, TypePtr type, Interpolation interpolation);
  /** Returns the declaration. */
  const VaryingPtr& decl() const { return decl_; }
  /** Writes the varying. Use it only in the vertex stage. */
  void set(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Reads the interpolated value. Use it only in the fragment stage. */
  Val value() const;

private:
  VaryingPtr decl_;
  Val read_;
};

/** Declares a varying. */
Varying varying(std::string name, TypePtr type,
                Interpolation interpolation = Interpolation::Perspective, Loc loc = Loc::current());

/** A fragment stage output (a color attachment). */
class Output {
public:
  /** Declares the output at location `index` (0 to 7). Prefer output() or blendOutput(). */
  Output(int index, TypePtr type, std::optional<int> blendSrc = std::nullopt);
  /** Returns the declaration. */
  const OutputPtr& decl() const { return decl_; }
  /** Writes the output. Use it only in the fragment stage. */
  void set(const ValueLike& value, Loc loc = Loc::current()) const;

private:
  OutputPtr decl_;
};

/**
 * Declares one of the two outputs of dual source blending: source 0 is the
 * color, source 1 the second blend factor input (src1 in the blend state).
 * Both write location 0, have the same type, and no other location output
 * can exist. WebGPU needs dual-source-blending; WebGL2 needs
 * WEBGL_blend_func_extended and GLSLOptions::blendFuncExtended.
 */
Output blendOutput(int source, TypePtr type = vec4f(), Loc loc = Loc::current());

/** Declares the fragment output at location `index` (0 to 7). */
Output output(int index, TypePtr type = vec4f(), Loc loc = Loc::current());

/**
 * A uniform block. Read a field with `block["fieldName"]`.
 * The layout is valid for std140 and for the WGSL uniform address space.
 */
class UniformBlock {
public:
  /** Wraps a uniform block resource. Use uniformBlock() to declare one. */
  explicit UniformBlock(UniformBlockPtr resource) : resource_(std::move(resource)) {}
  /** Returns the resource. Use it in AnalyzeOptions::retain. */
  const UniformBlockPtr& resource() const { return resource_; }
  /** Reads a field. Throws if the field does not exist. */
  Val operator[](std::string_view field) const;

private:
  UniformBlockPtr resource_;
};

/** Declares a uniform block. Do not use bool, mat2 or scalar arrays in it. */
UniformBlock uniformBlock(std::string name, const StructFields& fields,
                          ResourceFrequency frequency = ResourceFrequency::Material, Loc loc = Loc::current());

/** A texture, with a sampler when it is sampled. */
class Texture {
public:
  /** Wraps a texture resource. Use texture() to declare one. */
  explicit Texture(TexturePtr resource) : resource_(std::move(resource)) {}
  /** Returns the resource. */
  const TexturePtr& resource() const { return resource_; }

  /**
   * Samples with filtering at an explicit level of detail. The explicit level
   * makes the sample valid in nonuniform control flow on WGSL.
   */
  Val sample(const Val& coord, const ValueLike& level = 0,
             const std::optional<ValueLike>& layer = std::nullopt) const;
  /** Samples with screen space gradients. The gradients have the type of the coordinate. Not valid for depth textures. */
  Val sampleGrad(const Val& coord, const Val& ddx, const Val& ddy,
                 const std::optional<ValueLike>& layer = std::nullopt) const;
  /**
   * Samples at an explicit level with a constant texel offset. Use a vec2i
   * offset for 2D textures and a vec3i offset for 3D textures. Not valid for cube textures.
   */
  Val sampleOffset(const Val& coord, const Val& offset, const ValueLike& level = 0,
                   const std::optional<ValueLike>& layer = std::nullopt) const;
  /**
   * Compares `reference` with the depth at `coord` and returns the filtered
   * result in 0..1. Only for 2D depth textures declared with `comparison`.
   */
  Val compare(const Val& coord, const ValueLike& reference) const;
  /** Reads one texel without filtering. Use integer coordinates. Not valid for cube textures. */
  Val load(const Val& coord, const ValueLike& level = 0,
           const std::optional<ValueLike>& layer = std::nullopt) const;
  /** Returns the size in texels: (w, h) for 2D and cube, (w, h, d) for 3D, (w, h, layers) for arrays. */
  Val size(const ValueLike& level = 0) const;

private:
  TexturePtr resource_;
};

/** Options of texture(). */
struct TextureOptions {
  /** The texel type. Uint and Sint textures cannot be sampled; use load(). */
  TextureSampleType sampleType = TextureSampleType::Float;
  /** The update frequency. It selects the bind group. */
  ResourceFrequency frequency = ResourceFrequency::Material;
  /** Pairs a sampler. The default is true for float textures; integer textures cannot have one. */
  std::optional<bool> sampled;
  /**
   * Pairs a comparison sampler for compare(). Use it with a depth texture.
   * The texture is then only valid for compare().
   */
  bool comparison = false;
  /**
   * The texture holds r32float, rg32float or rgba32float texels and is
   * filtered. The device then needs the float32-filterable feature. Without
   * it, declare such textures with TextureSampleType::UnfilterableFloat.
   * The renderer chooses the format, so the DSL cannot check this.
   */
  bool float32Filterable = false;
};

/** Declares a texture. Invalid option combinations (for example a 3D depth texture) are an error. */
Texture texture(std::string name, TextureDimension dimension, TextureOptions options = {}, Loc loc = Loc::current());

/** One record of a table. */
class TableRecord {
public:
  /** Makes a record handle. Use Table::at() instead. */
  TableRecord(TablePtr table, ExprPtr index) : table_(std::move(table)), index_(std::move(index)) {}
  /** Reads a field of the record. Throws if the field does not exist. */
  Val operator[](std::string_view field) const;

private:
  TablePtr table_;
  ExprPtr index_;
};

/** Records indexed by a u32, for example one record per instance. */
class Table {
public:
  /** Wraps a table resource. Use table() to declare one. */
  explicit Table(TablePtr resource) : resource_(std::move(resource)) {}
  /** Returns the resource. */
  const TablePtr& resource() const { return resource_; }
  /** Returns the record at a u32 index. */
  TableRecord at(const ValueLike& index) const;

private:
  TablePtr resource_;
};

/**
 * Declares a table. Use only vec4f and mat4f fields. WGSL reads it as a read
 * only storage buffer; GLSL reads it as an RGBA32F texture.
 */
Table table(std::string name, const StructFields& fields,
            ResourceFrequency frequency = ResourceFrequency::Frame, Loc loc = Loc::current());

} // namespace moga
