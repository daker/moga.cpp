#pragma once

// Uniform layout valid for BOTH GLSL std140 and the WGSL uniform address
// space. Types whose rules differ are rejected (bool, mat2, arrays with
// element stride < 16); WGSL structs are emitted with explicit @align/@size
// from this layout so the two backends cannot drift.

#include "moga/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moga {

/** The place of one struct field in memory. */
struct FieldLayout {
  /** The field name. */
  std::string name;
  /** The field type. */
  TypePtr type;
  /** The byte offset from the start of the struct. */
  std::size_t offset = 0;
  /** The size in bytes. */
  std::size_t size = 0;
  /** The alignment in bytes. */
  std::size_t align = 0;
};

/** The uniform layout of a struct: valid for std140 and for the WGSL uniform address space. */
struct StructLayout {
  /**
   * The struct of this layout. Not owning: the struct owns its cached layout,
   * so an owning pointer here would keep both alive forever.
   */
  const TypeInfo* structType = nullptr;
  /** The size in bytes, a multiple of 16. */
  std::size_t size = 0;
  /** The alignment in bytes. */
  std::size_t align = 0;
  /** The fields, in declaration order. */
  std::vector<FieldLayout> fields;

  /** Returns the field with this name, or null. */
  const FieldLayout* find(std::string_view field) const;
};

/**
 * Returns the uniform layout of a struct. The layout is calculated one time
 * and kept by the struct. The returned pointer shares ownership of the struct,
 * so the layout stays valid while you hold it.
 */
std::shared_ptr<const StructLayout> structLayout(const TypePtr& s);

/**
 * Writes uniform values into a CPU buffer that follows a StructLayout.
 * Matrices are column major. The writer owns its buffer, or uses an external one.
 */
class UniformWriter {
public:
  /**
   * Makes a writer that owns a zero filled buffer of the layout size. The
   * writer shares ownership of the layout (and so of its struct).
   */
  explicit UniformWriter(std::shared_ptr<const StructLayout> layout);
  /** Makes a writer over an external buffer. The buffer must be at least the layout size. */
  UniformWriter(std::shared_ptr<const StructLayout> layout, std::span<std::byte> external);

  /**
   * A copy of a writer that owns its buffer gets its own copy of the bytes; a
   * copy of a writer over an external buffer writes the same external buffer.
   */
  UniformWriter(const UniformWriter& other);
  /** Copies a writer, as the copy constructor does. */
  UniformWriter& operator=(const UniformWriter& other);
  /** Moves a writer. The new writer keeps the buffer of the old one. */
  UniformWriter(UniformWriter&& other) noexcept;
  /** Moves a writer. This writer takes the buffer of the other one. */
  UniformWriter& operator=(UniformWriter&& other) noexcept;
  ~UniformWriter() = default;

  /** Returns the layout that the writer follows. */
  const StructLayout& layout() const { return *layout_; }
  /** Returns the layout's bytes of the buffer (an external buffer may be larger). Upload these. */
  std::span<const std::byte> bytes() const { return std::span<const std::byte>(data_).first(layout_->size); }

  /** Returns the byte offset of a field. Throws if the field does not exist. */
  std::size_t offsetOf(std::string_view field) const;

  // Each setter throws unless the field exists and has the type that it writes.

  /** Writes an f32 field, or an f16 field rounded to the nearest f16. Throws for another type. */
  void setFloat(std::string_view field, float v);
  /** Writes a uint field. */
  void setUint(std::string_view field, std::uint32_t v);
  /** Writes an int field. */
  void setInt(std::string_view field, std::int32_t v);
  /**
   * Writes the first components of a float vector field. An f16 vector gets
   * each value rounded to the nearest f16. Throws for more values than components.
   */
  void setVec(std::string_view field, std::span<const float> v);
  /** Writes the first components of an i32 vector field. Throws for more values than components. */
  void setIVec(std::string_view field, std::span<const std::int32_t> v);
  /** Writes the first components of a u32 vector field. Throws for more values than components. */
  void setUVec(std::string_view field, std::span<const std::uint32_t> v);
  /** Writes a vec4 field. */
  void setVec4(std::string_view field, float x, float y, float z, float w);
  /** Writes a column major 4x4 matrix. */
  void setMat4(std::string_view field, std::span<const float, 16> m);
  /** Writes a column major 3x3 matrix. Each column gets one padding float. */
  void setMat3(std::string_view field, std::span<const float, 9> m);
  /** Writes one element of a vec4 array field. Throws for an index past the end. */
  void setArrayVec4(std::string_view field, std::size_t index, float x, float y, float z,
                    float w);

private:
  /** Returns the field. Throws if it does not exist or does not have this type. */
  const FieldLayout& vectorField(std::string_view field, ScalarKind kind, std::size_t count) const;
  const FieldLayout& typedField(std::string_view field, const TypePtr& type) const;
  /** Returns true when the writer views its own buffer. */
  bool ownsBuffer() const { return !owned_.empty() && data_.data() == owned_.data(); }
  /** Copies bytes. Throws if the copy does not fit in the buffer. */
  void write(std::size_t byteOffset, const void* src, std::size_t n);

  std::shared_ptr<const StructLayout> layout_;
  std::vector<std::byte> owned_;
  std::span<std::byte> data_;
};

/**
 * Layout of a type in the WGSL storage address space. Arrays and structs are
 * not padded to 16 bytes as in uniform blocks: a vec3 aligns to 16 and has a
 * size of 12, a struct aligns to its largest field.
 */
struct StorageLayout {
  /** The size in bytes. */
  std::size_t size = 0;
  /** The alignment in bytes. */
  std::size_t align = 0;
  /** Distance between two array elements of this type. */
  std::size_t stride = 0;
  /** Struct fields; empty for other types. */
  std::vector<FieldLayout> fields;
};

/** Returns the storage layout of a host shareable type. Throws for bool. */
StorageLayout storageLayout(const TypePtr& t);

/** Texel offset of each field in a table record, by field name. */
using TableFieldOffsets = std::unordered_map<std::string, int>;
/** Returns the texel offset of each field. A vec4 uses 1 texel, a mat4 uses 4 texels. */
TableFieldOffsets tableFieldOffsets(const TypePtr& s);

} // namespace moga
