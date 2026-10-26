#include "moga/Layout.hpp"

#include "Half.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <string_view>

namespace moga {

namespace {

struct SizeAlign {
  std::size_t size;
  std::size_t align;
};

std::size_t roundUp(std::size_t n, std::size_t a) { return (n + a - 1) / a * a; }

/** offset + size, or an error when the struct does not fit size_t. */
std::size_t addBytes(std::size_t offset, std::size_t size, std::string_view where) {
  // room for the 16 byte rounding that follows
  if (size > std::numeric_limits<std::size_t>::max() - 16 - offset)
    throw ShaderError(ErrorCode::Layout, std::format("{}: the struct size does not fit in memory", where));
  return offset + size;
}

/** stride * length, or an error when the size does not fit size_t. */
std::size_t arrayBytes(std::size_t stride, int length, std::string_view where) {
  const auto n = static_cast<std::size_t>(length);
  if (stride != 0 && n > std::numeric_limits<std::size_t>::max() / stride)
    throw ShaderError(ErrorCode::Layout, std::format("{}: the array size {} x {} bytes does not fit in memory", where, n, stride));
  return stride * n;
}

SizeAlign sizeAlign(const TypePtr& t, const std::string& where);

/** f16 sizes and alignments are those of WGSL: vec2h 4, vec3h 6 aligned to 8, vec4h 8. */
SizeAlign halfSizeAlign(const TypePtr& t) {
  if (t->kind == TypeKind::Scalar) return {2, 2};
  const std::size_t align = t->size == 2 ? 4 : 8;
  if (t->kind == TypeKind::Vector) return {2u * static_cast<std::size_t>(t->size), align};
  // a matrix is an array of column vectors
  const std::size_t column = roundUp(2u * static_cast<std::size_t>(t->rows), t->rows == 2 ? 4 : 8);
  return {column * static_cast<std::size_t>(t->cols), t->rows == 2 ? 4u : 8u};
}

SizeAlign vectorSizeAlign(const TypePtr& t, const std::string& where) {
  if (t->scalar == ScalarKind::Bool)
    throw ShaderError(ErrorCode::Layout, std::format("{}: bool vectors are not host-shareable", where));
  if (t->scalar == ScalarKind::F16) return halfSizeAlign(t);
  if (t->size == 2) return {8, 8};
  return {t->size == 3 ? 12u : 16u, 16};
}

SizeAlign arraySizeAlign(const TypePtr& t, const std::string& where) {
  const SizeAlign e = sizeAlign(t->element, where);
  const std::size_t stride = roundUp(e.size, e.align);
  if (stride % 16 != 0) {
    throw ShaderError(ErrorCode::Layout, std::format(
        "{}: array<{}> stride {} is not a multiple of 16; use vec4 elements", where,
        typeName(t->element), stride));
  }
  return {arrayBytes(stride, t->length, where), 16};
}

SizeAlign sizeAlign(const TypePtr& t, const std::string& where) {
  switch (t->kind) {
  case TypeKind::Scalar:
    if (t->scalar == ScalarKind::Bool)
      throw ShaderError(ErrorCode::Layout, std::format("{}: bool is not host-shareable; use u32", where));
    return t->scalar == ScalarKind::F16 ? halfSizeAlign(t) : SizeAlign{4, 4};
  case TypeKind::Vector:
    return vectorSizeAlign(t, where);
  case TypeKind::Matrix:
    // f16 is WebGPU only, so its matrices follow WGSL alone
    if (t->scalar == ScalarKind::F16) return halfSizeAlign(t);
    if (t->rows != 4 && t->rows != 3)
      throw ShaderError(ErrorCode::Layout, 
          std::format("{}: {} layouts differ between std140 and WGSL", where, typeName(t)));
    return {16u * static_cast<std::size_t>(t->cols), 16};
  case TypeKind::Array:
    return arraySizeAlign(t, where);
  case TypeKind::Struct: {
    const auto l = structLayout(t);
    return {l->size, l->align};
  }
  case TypeKind::Atomic:
    throw ShaderError(ErrorCode::Layout, std::format("{}: atomic types are only valid in storage", where));
  case TypeKind::Void:
    break;
  }
  throw ShaderError(ErrorCode::Layout, std::format("{}: void has no layout", where));
}

std::shared_ptr<const StructLayout> computeLayout(const TypePtr& s) {
  auto layout = std::make_shared<StructLayout>();
  layout->structType = s.get();
  std::size_t offset = 0;
  std::size_t align = 16; // structs are 16 aligned in std140 and in WGSL uniform space
  for (const StructField& f : s->fields) {
    const SizeAlign sa = sizeAlign(f.type, std::format("{}.{}", s->name, f.name));
    offset = roundUp(offset, sa.align);
    layout->fields.push_back({f.name, f.type, offset, sa.size, sa.align});
    offset = addBytes(offset, sa.size, s->name);
    // std140: a member following a struct/array starts on a 16 byte boundary
    if (f.type->kind == TypeKind::Struct || f.type->kind == TypeKind::Array)
      offset = roundUp(offset, 16);
    align = std::max(align, sa.align);
  }
  layout->size = roundUp(std::max<std::size_t>(offset, 16), 16);
  layout->align = align;
  return layout;
}

} // namespace

const FieldLayout* StructLayout::find(std::string_view field) const {
  for (const FieldLayout& f : fields)
    if (f.name == field) return &f;
  return nullptr;
}

std::shared_ptr<const StructLayout> structLayout(const TypePtr& s) {
  if (s->kind != TypeKind::Struct)
    throw ShaderError(ErrorCode::InvalidType, std::format("{} is not a struct", typeName(s)));
  std::call_once(s->layoutOnce, [&] { s->layout = computeLayout(s); });
  // aliasing: owns the struct, points at its layout
  return std::shared_ptr<const StructLayout>(s, s->layout.get());
}

UniformWriter::UniformWriter(std::shared_ptr<const StructLayout> layout)
    : layout_(std::move(layout)) {
  if (!layout_) throw ShaderError(ErrorCode::UniformWriter, "UniformWriter needs a layout");
  owned_.resize(layout_->size);
  data_ = owned_;
}

UniformWriter::UniformWriter(std::shared_ptr<const StructLayout> layout,
                             std::span<std::byte> external)
    : layout_(std::move(layout)), data_(external) {
  if (!layout_) throw ShaderError(ErrorCode::UniformWriter, "UniformWriter needs a layout");
  if (external.size() < layout_->size)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}: buffer of {} bytes is smaller than the {} byte layout",
                                  layout_->structType->name, external.size(), layout_->size));
}

UniformWriter::UniformWriter(const UniformWriter& other)
    : layout_(other.layout_), owned_(other.owned_) {
  data_ = other.ownsBuffer() ? std::span<std::byte>(owned_) : other.data_;
}

UniformWriter& UniformWriter::operator=(const UniformWriter& other) {
  if (this != &other) {
    const bool owns = other.ownsBuffer();
    layout_ = other.layout_;
    owned_ = other.owned_;
    data_ = owns ? std::span<std::byte>(owned_) : other.data_;
  }
  return *this;
}

UniformWriter::UniformWriter(UniformWriter&& other) noexcept
    : layout_(std::move(other.layout_)), owned_(std::move(other.owned_)), data_(other.data_) {
  // a moved vector keeps its buffer, so the view stays valid; the source views nothing
  other.data_ = {};
}

UniformWriter& UniformWriter::operator=(UniformWriter&& other) noexcept {
  if (this != &other) {
    layout_ = std::move(other.layout_);
    owned_ = std::move(other.owned_);
    data_ = other.data_;
    other.data_ = {};
  }
  return *this;
}

std::size_t UniformWriter::offsetOf(std::string_view field) const {
  const FieldLayout* f = layout_->find(field);
  if (f == nullptr)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{} has no field {}", layout_->structType->name, field));
  return f->offset;
}

const FieldLayout& UniformWriter::typedField(std::string_view field, const TypePtr& type) const {
  const FieldLayout* f = layout_->find(field);
  if (f == nullptr)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{} has no field {}", layout_->structType->name, field));
  if (!typesEqual(f->type, type))
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{} is {}, not {}", layout_->structType->name, field,
                                  typeName(f->type), typeName(type)));
  return *f;
}

void UniformWriter::write(std::size_t byteOffset, const void* src, std::size_t n) {
  if (byteOffset > data_.size() || n > data_.size() - byteOffset)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}: write of {} bytes at {} is past the {} byte buffer",
                                  layout_->structType->name, n, byteOffset, data_.size()));
  std::memcpy(data_.data() + byteOffset, src, n);
}

void UniformWriter::setFloat(std::string_view field, float v) {
  const FieldLayout* f = layout_->find(field);
  if (f != nullptr && f->type == f16()) {
    const std::uint16_t bits = detail::halfBits(v);
    write(f->offset, &bits, 2);
    return;
  }
  write(typedField(field, f32()).offset, &v, 4);
}
void UniformWriter::setUint(std::string_view field, std::uint32_t v) {
  write(typedField(field, u32()).offset, &v, 4);
}
void UniformWriter::setInt(std::string_view field, std::int32_t v) {
  write(typedField(field, i32()).offset, &v, 4);
}

void UniformWriter::setVec(std::string_view field, std::span<const float> v) {
  const FieldLayout* f = layout_->find(field);
  if (f == nullptr)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{} has no field {}", layout_->structType->name, field));
  if (f->type->kind != TypeKind::Vector || !isFloatKind(f->type->scalar))
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{} is {}, not a float vector", layout_->structType->name,
                                  field, typeName(f->type)));
  if (v.size() > static_cast<std::size_t>(f->type->size))
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{} has {} components, got {} values",
                                  layout_->structType->name, field, f->type->size, v.size()));
  if (f->type->scalar == ScalarKind::F16) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      const std::uint16_t bits = detail::halfBits(v[i]);
      write(f->offset + i * 2, &bits, 2);
    }
    return;
  }
  write(f->offset, v.data(), v.size() * 4);
}

/** The field, after checking that it is a vector of this kind with room for `count` components. */
const FieldLayout& UniformWriter::vectorField(std::string_view field, ScalarKind kind, std::size_t count) const {
  const FieldLayout* f = layout_->find(field);
  if (f == nullptr)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{} has no field {}", layout_->structType->name, field));
  if (f->type->kind != TypeKind::Vector || f->type->scalar != kind)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{} is {}, not a {} vector", layout_->structType->name, field,
                                  typeName(f->type), typeName(scalarType(kind))));
  if (count > static_cast<std::size_t>(f->type->size))
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{} has {} components, got {} values", layout_->structType->name, field,
                                  f->type->size, count));
  return *f;
}

void UniformWriter::setIVec(std::string_view field, std::span<const std::int32_t> v) {
  write(vectorField(field, ScalarKind::I32, v.size()).offset, v.data(), v.size_bytes());
}

void UniformWriter::setUVec(std::string_view field, std::span<const std::uint32_t> v) {
  write(vectorField(field, ScalarKind::U32, v.size()).offset, v.data(), v.size_bytes());
}

void UniformWriter::setVec4(std::string_view field, float x, float y, float z, float w) {
  const float v[4] = {x, y, z, w};
  write(typedField(field, vec4f()).offset, v, sizeof v);
}

void UniformWriter::setMat4(std::string_view field, std::span<const float, 16> m) {
  write(typedField(field, mat4f()).offset, m.data(), 64);
}

void UniformWriter::setMat3(std::string_view field, std::span<const float, 9> m) {
  const std::size_t o = typedField(field, mat3f()).offset;
  for (std::size_t c = 0; c < 3; ++c) {
    const float col[4] = {m[c * 3], m[c * 3 + 1], m[c * 3 + 2], 0.0f};
    write(o + c * 16, col, sizeof col);
  }
}

void UniformWriter::setArrayVec4(std::string_view field, std::size_t index, float x, float y,
                                 float z, float w) {
  const FieldLayout* f = layout_->find(field);
  if (f == nullptr)
    throw ShaderError(ErrorCode::UniformWriter, std::format("{} has no field {}", layout_->structType->name, field));
  if (f->type->kind != TypeKind::Array || f->type->element != vec4f())
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{} is {}, not a vec4f array", layout_->structType->name,
                                  field, typeName(f->type)));
  if (index >= static_cast<std::size_t>(f->type->length))
    throw ShaderError(ErrorCode::UniformWriter, std::format("{}.{}: index {} is past the {} elements",
                                  layout_->structType->name, field, index, f->type->length));
  const float v[4] = {x, y, z, w};
  write(f->offset + index * 16, v, sizeof v);
}

StorageLayout storageLayout(const TypePtr& t) {
  StorageLayout l;
  switch (t->kind) {
  case TypeKind::Scalar:
  case TypeKind::Atomic:
    if (t->scalar == ScalarKind::Bool)
      throw ShaderError(ErrorCode::Layout, "bool is not host-shareable; use u32 in storage");
    l.size = t->scalar == ScalarKind::F16 ? 2 : 4;
    l.align = l.size;
    break;
  case TypeKind::Vector:
    if (t->scalar == ScalarKind::Bool)
      throw ShaderError(ErrorCode::Layout, std::format("{} is not host-shareable", typeName(t)));
    if (t->scalar == ScalarKind::F16) {
      l.size = 2u * static_cast<std::size_t>(t->size);
      l.align = t->size == 2 ? 4 : 8;
      break;
    }
    l.size = 4u * static_cast<std::size_t>(t->size);
    l.align = t->size == 2 ? 8 : 16;
    break;
  case TypeKind::Matrix: {
    // a matrix is an array of column vectors
    const StorageLayout column = storageLayout(vectorType(t->scalar, t->rows));
    l.align = column.align;
    l.size = static_cast<std::size_t>(t->cols) * column.stride;
    break;
  }
  case TypeKind::Array: {
    const StorageLayout element = storageLayout(t->element);
    l.align = element.align;
    l.size = arrayBytes(element.stride, t->length, typeName(t));
    break;
  }
  case TypeKind::Struct: {
    std::size_t offset = 0;
    l.align = 1;
    for (const StructField& f : t->fields) {
      const StorageLayout fl = storageLayout(f.type);
      offset = roundUp(offset, fl.align);
      l.fields.push_back({f.name, f.type, offset, fl.size, fl.align});
      offset = addBytes(offset, fl.size, t->name);
      l.align = std::max(l.align, fl.align);
    }
    l.size = roundUp(offset, l.align);
    break;
  }
  case TypeKind::Void:
    throw ShaderError(ErrorCode::Layout, "void has no layout");
  }
  l.stride = roundUp(l.size, l.align);
  return l;
}

TableFieldOffsets tableFieldOffsets(const TypePtr& s) {
  TableFieldOffsets out;
  int texel = 0;
  for (const StructField& f : s->fields) {
    out[f.name] = texel;
    texel += f.type->kind == TypeKind::Matrix ? 4 : 1;
  }
  return out;
}

} // namespace moga
