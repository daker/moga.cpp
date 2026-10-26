#include "Context.hpp"
#include "Known.hpp"
#include "moga/dsl/Declarations.hpp"

#include <algorithm>
#include <format>

namespace moga {

using namespace detail;

namespace {

/** WGSL puts stage inputs and outputs at @location, which takes numeric scalars and vectors. */
void requireLocationType(const TypePtr& type, std::string_view what) {
  if (!isNumeric(type))
    throw ShaderError(ErrorCode::InvalidType, std::format("{} must be a numeric scalar or vector, got {}", what, typeName(type)));
}

} // namespace

Val attribute(std::string name, TypePtr type, VertexStep step, Loc loc) {
  detail::noteUserLocation(loc);
  return detail::guard([&]() -> Val {
    requireLocationType(type, std::format("Attribute {}", name));
    auto decl = std::make_shared<const AttributeDecl>(AttributeDecl{std::move(name), type, step});
    return Val(makeExpr(ex::Attribute{decl}, type));
  }, name, type, step);
}

Varying::Varying(std::string name, TypePtr type, Interpolation interpolation)
    : decl_(std::make_shared<const VaryingDecl>(VaryingDecl{name, type, interpolation})),
      read_(makeExpr(ex::Varying{decl_}, type)) {
  detail::check([&] {
    requireLocationType(type, std::format("Varying {}", name));
    if (isInteger(type) && interpolation != Interpolation::Flat &&
        interpolation != Interpolation::FlatEither)
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Integer varying \"{}\" must use flat interpolation", name));
  });
}

void Varying::set(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    requireStage(StageName::Vertex, std::format("Writing varying \"{}\"", decl_->name));
    const Val v = toVal(value, decl_->type);
    checkType(v, decl_->type, std::format("varying {}", decl_->name));
    pushStatement(st::SetVarying{decl_, v.node()}, loc);
  }, value, loc);
}

Val Varying::value() const {
  return detail::guard([&]() -> Val {
    requireStage(StageName::Fragment, std::format("Reading varying \"{}\"", decl_->name));
    return read_;
  });
}

Varying varying(std::string name, TypePtr type, Interpolation interpolation, Loc loc) {
  detail::noteUserLocation(loc);
  return Varying(std::move(name), std::move(type), interpolation);
}

Output::Output(int index, TypePtr type, std::optional<int> blendSrc)
    : decl_(std::make_shared<const OutputDecl>(OutputDecl{index, std::move(type), blendSrc})) {
  detail::check([&] {
    if (blendSrc && *blendSrc != 0 && *blendSrc != 1)
      throw ShaderError(ErrorCode::OutOfRange, std::format("Blend source {} is not 0 or 1", *blendSrc));
    // WebGPU has at most 8 color attachments
    if (index < 0 || index > 7) throw ShaderError(ErrorCode::OutOfRange, std::format("Output index {} is not in 0..7", index));
    requireLocationType(decl_->type, std::format("Output {}", index));
  });
}

void Output::set(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    requireStage(StageName::Fragment, "Writing a fragment output");
    const Val v = toVal(value, decl_->type);
    checkType(v, decl_->type, std::format("output {}", decl_->index));
    pushStatement(st::SetOutput{decl_, v.node()}, loc);
  }, value, loc);
}

Output output(int index, TypePtr type, Loc loc) {
  detail::noteUserLocation(loc);
  return Output(index, std::move(type));
}

Output blendOutput(int source, TypePtr type, Loc loc) {
  detail::noteUserLocation(loc);
  return Output(0, std::move(type), source);
}

static const StructField& findField(const TypePtr& s, std::string_view field,
                                    std::string_view owner) {
  for (const StructField& f : s->fields)
    if (f.name == field) return f;
  throw ShaderError(ErrorCode::UnknownName, std::format("{} has no field \"{}\"", owner, field));
}

Val UniformBlock::operator[](std::string_view field) const {
  return detail::guard([&]() -> Val {
    const StructField& f = findField(resource_->structType, field, resource_->name);
    return Val(makeExpr(ex::Uniform{resource_, f.name}, f.type));
  }, field);
}

UniformBlock uniformBlock(std::string name, const StructFields& fields,
                          ResourceFrequency frequency, Loc loc) {
  detail::noteUserLocation(loc);
  TypePtr s = makeStruct(name + "Block", fields);
  return UniformBlock(
      std::make_shared<const UniformBlockResource>(std::move(name), std::move(s), frequency));
}

static TypePtr texelType(const TextureResource& r) {
  switch (r.sampleType) {
  case TextureSampleType::Uint:
    return vec4u();
  case TextureSampleType::Sint:
    return vec4i();
  case TextureSampleType::Depth:
    return f32();
  default:
    return vec4f();
  }
}

namespace {

const TypePtr& coordType(const TextureResource& r) {
  return r.dimension == TextureDimension::D3 || r.dimension == TextureDimension::Cube ? vec3f()
                                                                                      : vec2f();
}

/** Throws unless the texture has a plain (not comparison) sampler. */
bool isIntegerTexture(TextureSampleType t) {
  return t == TextureSampleType::Uint || t == TextureSampleType::Sint;
}

void requireFilterSampler(const TextureResource& r) {
  // WGSL has no textureSampleLevel overload for integer textures
  if (isIntegerTexture(r.sampleType))
    throw ShaderError(ErrorCode::TextureUsage, std::format("Texture {} has integer texels; use load()", r.name));
  if (!r.sampled) throw ShaderError(ErrorCode::TextureUsage, std::format("Texture {} has no sampler", r.name));
  if (r.comparison)
    throw ShaderError(ErrorCode::TextureUsage, std::format("Texture {} has a comparison sampler; use compare()", r.name));
}

/** Returns the layer, which a 2D array texture needs and other textures must not have. */
ExprPtr layerNode(const TextureResource& r, const std::optional<ValueLike>& layer) {
  const bool array = r.dimension == TextureDimension::D2Array;
  if (array && !layer)
    throw ShaderError(ErrorCode::TextureUsage, std::format("{}: a 2D array texture needs a layer", r.name));
  if (!array && layer)
    throw ShaderError(ErrorCode::TextureUsage, std::format("{}: only 2D array textures take a layer", r.name));
  if (!layer) return nullptr;
  const Val l = toVal(*layer, i32());
  if (l.type() != i32() && l.type() != u32())
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{}: the layer must be i32 or u32, got {}", r.name, typeName(l.type())));
  return l.node();
}

/**
 * Returns the level of a filtered sample. A depth texture takes an integer
 * level (WGSL has no float level overload for depth); others take f32.
 */
ExprPtr levelNode(const TextureResource& r, const ValueLike& level) {
  if (r.sampleType != TextureSampleType::Depth) {
    // toVal() keeps an existing integer Val, so check the type
    const Val l = toVal(level, f32());
    checkType(l, f32(), std::format("{} level", r.name));
    return l.node();
  }
  const Val l = toVal(level, i32());
  if (l.type() != i32() && l.type() != u32())
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{}: a depth texture takes an i32 or u32 level, got {}", r.name,
                                  typeName(l.type())));
  return l.node();
}

} // namespace

Val Texture::sample(const Val& coord, const ValueLike& level,
                    const std::optional<ValueLike>& layer) const {
  return detail::guard([&]() -> Val {
    const TextureResource& r = *resource_;
    requireFilterSampler(r);
    checkType(coord, coordType(r), std::format("{} coordinate", r.name));
    // a depth sample gives one f32 on both backends
    return Val(makeExpr(ex::Sample{resource_, coord.node(), levelNode(r, level), layerNode(r, layer)},
                        texelType(r)));
  }, coord, level, layer);
}

Val Texture::sampleGrad(const Val& coord, const Val& ddx, const Val& ddy,
                        const std::optional<ValueLike>& layer) const {
  return detail::guard([&]() -> Val {
    const TextureResource& r = *resource_;
    requireFilterSampler(r);
    if (r.sampleType == TextureSampleType::Depth)
      throw ShaderError(ErrorCode::TextureUsage, std::format("{}: WGSL has no gradient sample for depth textures", r.name));
    checkType(coord, coordType(r), std::format("{} coordinate", r.name));
    checkType(ddx, coordType(r), std::format("{} ddx", r.name));
    checkType(ddy, coordType(r), std::format("{} ddy", r.name));
    ex::Sample s{resource_, coord.node(), nullptr, layerNode(r, layer)};
    s.ddx = ddx.node();
    s.ddy = ddy.node();
    return Val(makeExpr(std::move(s), texelType(r)));
  }, coord, ddx, ddy, layer);
}

Val Texture::sampleOffset(const Val& coord, const Val& offset, const ValueLike& level,
                          const std::optional<ValueLike>& layer) const {
  return detail::guard([&]() -> Val {
    const TextureResource& r = *resource_;
    requireFilterSampler(r);
    if (r.dimension == TextureDimension::Cube)
      throw ShaderError(ErrorCode::TextureUsage, std::format("{}: cube textures cannot use a texel offset", r.name));
    checkType(coord, coordType(r), std::format("{} coordinate", r.name));
    checkType(offset, r.dimension == TextureDimension::D3 ? vec3i() : vec2i(),
              std::format("{} offset", r.name));
    // both languages need a constant expression; check the value the shader converts it to
    if (!isConstExpression(*offset.node()))
      throw ShaderError(ErrorCode::ConstantRequired, std::format("{}: the texel offset must be a constant", r.name));
    // the range check is required, so a constant that the DSL cannot evaluate is rejected too
    const auto known = knownRanges(*offset.node());
    if (!known)
      throw ShaderError(ErrorCode::ConstantRequired, std::format("{}: the texel offset must be a constant the DSL can evaluate (literals, named "
                                    "constants, arithmetic, bit operations, min, max, clamp, abs, constructors)",
                                    r.name));
    for (const KnownRange& c : *known)
      if (c.low < -8 || c.high > 7)
        throw ShaderError(ErrorCode::OutOfRange, std::format("{}: texel offset component {} is not in -8..7", r.name, describe(c)));
    ex::Sample s{resource_, coord.node(), levelNode(r, level), layerNode(r, layer)};
    s.offset = offset.node();
    return Val(makeExpr(std::move(s), texelType(r)));
  }, coord, offset, level, layer);
}

Val Texture::compare(const Val& coord, const ValueLike& reference) const {
  return detail::guard([&]() -> Val {
    const TextureResource& r = *resource_;
    if (!r.comparison)
      throw ShaderError(ErrorCode::TextureUsage, std::format("Texture {} has no comparison sampler", r.name));
    checkType(coord, vec2f(), std::format("{} coordinate", r.name));
    ex::Sample s{resource_, coord.node(), nullptr, nullptr};
    const Val ref = toVal(reference, f32());
    checkType(ref, f32(), std::format("{} depth reference", r.name));
    s.depthRef = ref.node();
    return Val(makeExpr(std::move(s), f32()));
  }, coord, reference);
}

Val Texture::load(const Val& coord, const ValueLike& level,
                  const std::optional<ValueLike>& layer) const {
  return detail::guard([&]() -> Val {
    const TextureResource& r = *resource_;
    if (r.comparison)
      throw ShaderError(ErrorCode::TextureUsage, std::format("Texture {} has a comparison sampler; use compare()", r.name));
    if (r.dimension == TextureDimension::Cube)
      throw ShaderError(ErrorCode::TextureUsage, std::format("{}: cube textures cannot load texels", r.name));
    const int want = r.dimension == TextureDimension::D3 ? 3 : 2;
    const TypePtr& ct = coord.type();
    if (ct->kind != TypeKind::Vector || ct->size != want || ct->scalar != ScalarKind::I32)
      throw ShaderError(ErrorCode::TypeMismatch, std::format("{}.load needs vec{}i coordinates, got {}", r.name, want,
                                    typeName(ct)));
    return Val(makeExpr(ex::Load{resource_, coord.node(), integerScalar(level, std::format("{} level", r.name)).node(),
                                 layerNode(r, layer)},
                        texelType(r)));
  }, coord, level, layer);
}

Val Texture::size(const ValueLike& level) const {
  return detail::guard([&]() -> Val {
    const TextureDimension d = resource_->dimension;
    const int n = d == TextureDimension::D2 || d == TextureDimension::Cube ? 2 : 3;
    return Val(makeExpr(ex::TexSize{resource_, integerScalar(level, std::format("{} level", resource_->name)).node()},
                        vectorType(ScalarKind::I32, n)));
  }, level);
}

Texture texture(std::string name, TextureDimension dimension, TextureOptions options, Loc loc) {
  detail::noteUserLocation(loc);
  detail::check([&] {
    if (options.sampleType == TextureSampleType::Depth && dimension == TextureDimension::D3)
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Texture {}: there are no 3D depth textures", name));
    if (options.comparison) {
      if (options.sampleType != TextureSampleType::Depth || dimension != TextureDimension::D2)
        throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Texture {}: a comparison sampler needs a 2D depth texture", name));
      if (options.sampled == false)
        throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Texture {}: a comparison texture needs its sampler", name));
    }
    if (options.float32Filterable &&
        (options.sampleType != TextureSampleType::Float || options.sampled == false))
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Texture {}: float32Filterable needs a sampled float texture", name));
    if (options.sampled == true && isIntegerTexture(options.sampleType))
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Texture {}: integer textures cannot be sampled; use load()", name));
  });
  const bool sampled = options.comparison ||
                       options.sampled.value_or(options.sampleType == TextureSampleType::Float);
  return Texture(std::make_shared<const TextureResource>(std::move(name), dimension,
                                                         options.sampleType, options.frequency,
                                                         sampled, options.comparison,
                                                         options.float32Filterable));
}

Val uniform(std::string name, TypePtr type, ResourceFrequency frequency, Loc loc) {
  detail::noteUserLocation(loc);
  return detail::guard([&]() -> Val {
    auto decl = std::make_shared<const UniformDecl>(UniformDecl{std::move(name), type, frequency});
    return Val(makeExpr(ex::LooseUniform{decl}, type));
  }, name, type, frequency);
}

Val TableRecord::operator[](std::string_view field) const {
  return detail::guard([&]() -> Val {
    const StructField& f = findField(table_->structType, field, table_->name);
    return Val(makeExpr(ex::TableRead{table_, index_, f.name}, f.type));
  }, field);
}

TableRecord Table::at(const ValueLike& index) const {
  const Val idx = detail::guard([&]() -> Val { return toVal(index, u32()); }, index);
  if (idx.type() != u32() && !detail::poisoned(idx))
    detail::fail(ShaderError(ErrorCode::TypeMismatch, std::format("Table index must be u32, got {}", typeName(idx.type()))));
  return TableRecord(resource_, idx.node());
}

Table table(std::string name, const StructFields& fields, ResourceFrequency frequency, Loc loc) {
  detail::noteUserLocation(loc);
  int stride = 0;
  for (const auto& [field, t] : fields) {
    if (t == vec4f()) stride += 1;
    else if (t == mat4f()) stride += 4;
    else
      detail::fail(ShaderError(ErrorCode::InvalidType, std::format("Table {}.{}: only vec4f/mat4f fields are allowed, got {}",
                                                                 name, field, typeName(t))));
  }
  TypePtr s = makeStruct(name + "Record", fields);
  return Table(
      std::make_shared<const TableResource>(std::move(name), std::move(s), frequency, stride));
}

} // namespace moga
