#include "BuiltinTable.hpp"
#include "../Steps.hpp"
#include "Context.hpp"
#include "Known.hpp"
#include "moga/dsl/Compute.hpp"
#include "moga/dsl/Statements.hpp"
#include "moga/Capabilities.hpp"
#include "moga/Layout.hpp"

#include <array>
#include <bit>
#include <format>

namespace moga {

using namespace detail;

namespace {

/** Throws in the vertex stage, which cannot write storage. */
void requireNotVertex(std::string_view what) {
  if (currentStage() == StageName::Vertex)
    throw ShaderError(ErrorCode::WrongStage, std::format("{} is not valid in the vertex stage", what));
}

/** Throws outside the compute and fragment stages. */
void requireSubgroupStage(std::string_view what) {
  if (buildingDetachedFunction()) return;
  const auto stage = currentStage();
  if (stage != StageName::Compute && stage != StageName::Fragment)
    throw ShaderError(ErrorCode::WrongStage, std::format("{} is only valid in the compute and fragment stages", what));
}

void requireIndex(const Val& i, std::string_view what) {
  if (i.type() != u32() && i.type() != i32())
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{} index must be u32 or i32, got {}", what, typeName(i.type())));
}

/** The storage element or workgroup array that an lvalue expression writes. */
const Expr& rootOf(const Expr& e) {
  if (const auto* m = std::get_if<ex::Member>(&e.node)) return rootOf(*m->base);
  if (const auto* i = std::get_if<ex::Index>(&e.node)) return rootOf(*i->base);
  if (const auto* s = std::get_if<ex::Swizzle>(&e.node)) return rootOf(*s->base);
  return e;
}

/** Throws unless the target is an atomic in a read/write buffer or a workgroup array. */
void requireAtomicTarget(const Val& target, std::string_view what) {
  if (target.type()->kind != TypeKind::Atomic)
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{} needs an atomic target, got {}", what, typeName(target.type())));
  const Expr& root = rootOf(*target.node());
  if (const auto* s = std::get_if<ex::StorageElement>(&root.node)) {
    if (s->buffer->access != StorageAccess::ReadWrite)
      throw ShaderError(ErrorCode::NotAssignable, std::format("{}: storage {} is read only", what, s->buffer->name));
    return;
  }
  if (std::holds_alternative<ex::Workgroup>(root.node)) return;
  throw ShaderError(ErrorCode::NotAssignable, std::format("{}: the target must be in a storage buffer or a workgroup array", what));
}

Val callBuiltin(BuiltinFunction fn, TypePtr type, const std::vector<Val>& args) {
  std::vector<ExprPtr> nodes;
  for (const Val& a : args) nodes.push_back(a.node());
  return Val(makeExpr(ex::Call{fn, std::move(nodes)}, std::move(type)));
}

Val atomicOp(AtomicOp op, const Val& target, const ValueLike& value, const Loc& loc) {
  const std::string_view what = atomicOpName(op);
  requireAtomicTarget(target, what);
  requireNotVertex(what);
  const TypePtr& scalar = scalarType(target.type()->scalar);
  const Val v = toVal(value, scalar);
  checkType(v, scalar, what);
  // the old value goes in a constant, so the operation runs once whether or not it is used
  return Let(Val(makeExpr(ex::Atomic{op, target.node(), v.node()}, scalar)), "old", loc);
}

const TypePtr& texelValueType(const StorageTextureResource& r) {
  return vectorType(texelFormatScalar(r.format), 4);
}

void checkTexelCoord(const StorageTextureResource& r, const Val& coord,
                     const std::optional<ValueLike>& layer) {
  const int want = r.dimension == TextureDimension::D3 ? 3 : 2;
  const TypePtr& t = coord.type();
  if (t->kind != TypeKind::Vector || t->size != want ||
      (t->scalar != ScalarKind::I32 && t->scalar != ScalarKind::U32))
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{} needs vec{}i or vec{}u coordinates, got {}", r.name, want,
                                  want, typeName(t)));
  if ((r.dimension == TextureDimension::D2Array) != layer.has_value())
    throw ShaderError(ErrorCode::TextureUsage, std::format("{}: give a layer for 2D array textures only", r.name));
}

/** WGSL subgroups have at most 128 invocations; a constant id must be below that. */
constexpr double kMaxSubgroupSize = 128;

void requireSubgroupId(double id, std::string_view what) {
  if (!(id >= 0 && id < kMaxSubgroupSize))
    throw ShaderError(ErrorCode::OutOfRange, std::format("{}: the id {} is not in 0..{}", what, id, kMaxSubgroupSize - 1));
}

/** Checks the stage and the first argument against the row of a subgroup builtin, then makes the call. */
Val subgroupCall(BuiltinFunction fn, const std::vector<Val>& args) {
  requireSubgroupStage(builtinName(fn));
  return callBuiltin(fn, checkBuiltin(fn, args.empty() ? nullptr : args.front().type()), args);
}

Val builtin(BuiltinInput name, const TypePtr& type) {
  return Val(makeExpr(ex::Builtin{name}, type));
}

Val computeBuiltin(BuiltinInput name, const TypePtr& type) {
  requireStage(StageName::Compute, std::format("Builtin {}", builtinInputName(name)));
  return builtin(name, type);
}

} // namespace

// ------------------------------------------------------------------ storage

Val StorageBuffer::operator[](const ValueLike& index) const {
  return detail::guard([&]() -> Val {
    const Val i = toVal(index, u32());
    requireIndex(i, resource_->name);
    // WGSL rejects a negative const expression index; the upper bound is the bound buffer's
    if (const auto known = knownRanges(*i.node()); known && known->front().low < 0)
      throw ShaderError(ErrorCode::OutOfRange, std::format("{}: index {} is negative", resource_->name, describe(known->front())));
    return Val(makeExpr(ex::StorageElement{resource_, i.node()}, resource_->elementType));
  }, index);
}

Val StorageBuffer::length() const {
  return detail::guard([&]() -> Val {
    return Val(makeExpr(ex::StorageLength{resource_}, u32()));
  });
}

StorageBuffer storage(std::string name, TypePtr elementType, StorageAccess access,
                      ResourceFrequency frequency, Loc loc) {
  detail::noteUserLocation(loc);
  detail::check([&] {
    if (access == StorageAccess::Write)
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Storage {}: buffers take Read or ReadWrite access", name));
    storageLayout(elementType); // throws for types that are not host shareable
    if (access == StorageAccess::Read && containsAtomic(elementType))
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Storage {}: atomics need ReadWrite access", name));
  });
  return StorageBuffer(std::make_shared<const StorageBufferResource>(
      std::move(name), std::move(elementType), access, frequency));
}

void StorageTexture::store(const Val& coord, const ValueLike& value,
                           const std::optional<ValueLike>& layer, Loc loc) const {
  return detail::guard([&] {
    const StorageTextureResource& r = *resource_;
    if (r.access == StorageAccess::Read)
      throw ShaderError(ErrorCode::NotAssignable, std::format("Storage texture {} is read only", r.name));
    requireNotVertex(std::format("Writing storage texture {}", r.name));
    checkTexelCoord(r, coord, layer);
    const Val v = toVal(value, texelValueType(r));
    checkType(v, texelValueType(r), std::format("{} value", r.name));
    pushStatement(st::TextureStore{resource_, coord.node(), layer ? integerScalar(*layer, std::format("{} layer", r.name)).node() : nullptr,
                                   v.node()},
                  loc);
  }, coord, value, layer, loc);
}

Val StorageTexture::load(const Val& coord, const std::optional<ValueLike>& layer) const {
  return detail::guard([&]() -> Val {
    const StorageTextureResource& r = *resource_;
    if (r.access == StorageAccess::Write)
      throw ShaderError(ErrorCode::TextureUsage, std::format("Storage texture {} is write only", r.name));
    checkTexelCoord(r, coord, layer);
    return Val(makeExpr(ex::StorageLoad{resource_, coord.node(), layer ? integerScalar(*layer, std::format("{} layer", r.name)).node() : nullptr},
                        texelValueType(r)));
  }, coord, layer);
}

Val StorageTexture::size() const {
  return detail::guard([&]() -> Val {
    const int n = resource_->dimension == TextureDimension::D2 ? 2 : 3;
    return Val(makeExpr(ex::StorageSize{resource_}, vectorType(ScalarKind::U32, n)));
  });
}

StorageTexture storageTexture(std::string name, TextureDimension dimension, TexelFormat format,
                              StorageAccess access, ResourceFrequency frequency, Loc loc) {
  detail::noteUserLocation(loc);
  detail::check([&] {
    if (dimension == TextureDimension::Cube)
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Storage texture {}: cube textures are not valid", name));
    if (!storageFormatFeatures(format, access)) {
      static constexpr std::array<std::string_view, 3> kAccess{"read", "write", "read/write"};
      throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Storage texture {}: WebGPU has no {} {} storage texture", name,
                                    kAccess[static_cast<std::size_t>(access)], texelFormatName(format)));
    }
  });
  return StorageTexture(std::make_shared<const StorageTextureResource>(std::move(name), dimension,
                                                                       format, access, frequency));
}

Val workgroupArray(std::string name, TypePtr elementType, int length, Loc loc) {
  detail::noteUserLocation(loc);
  return detail::guard([&]() -> Val {
    if (length <= 0) throw ShaderError(ErrorCode::InvalidDeclaration, std::format("Workgroup array {}: length must be positive", name));
    const TypePtr type = arrayOf(std::move(elementType), length);
    auto decl = std::make_shared<const WorkgroupDecl>(WorkgroupDecl{std::move(name), type});
    return Val(makeExpr(ex::Workgroup{decl}, type));
  }, name, elementType, length);
}

// ------------------------------------------------------------------ builtins

Val globalId() {
  return detail::guard([&]() -> Val { return computeBuiltin(BuiltinInput::GlobalId, vec3u()); });
}
Val localId() {
  return detail::guard([&]() -> Val { return computeBuiltin(BuiltinInput::LocalId, vec3u()); });
}
Val workgroupId() {
  return detail::guard([&]() -> Val { return computeBuiltin(BuiltinInput::WorkgroupId, vec3u()); });
}
Val numWorkgroups() {
  return detail::guard([&]() -> Val { return computeBuiltin(BuiltinInput::NumWorkgroups, vec3u()); });
}
Val localIndex() {
  return detail::guard([&]() -> Val { return computeBuiltin(BuiltinInput::LocalIndex, u32()); });
}

Val subgroupSize() {
  return detail::guard([&]() -> Val {
    requireSubgroupStage("Builtin subgroupSize");
    return builtin(BuiltinInput::SubgroupSize, u32());
  });
}

Val subgroupInvocationId() {
  return detail::guard([&]() -> Val {
    requireSubgroupStage("Builtin subgroupInvocationId");
    return builtin(BuiltinInput::SubgroupInvocationId, u32());
  });
}

// ------------------------------------------------------------------ synchronization

void workgroupBarrier(Loc loc) {
  return detail::guard([&] {
    if (!buildingDetachedFunction()) requireStage(StageName::Compute, "workgroupBarrier()");
    pushStatement(st::Barrier{BarrierKind::Workgroup}, loc);
  }, loc);
}

void storageBarrier(Loc loc) {
  return detail::guard([&] {
    if (!buildingDetachedFunction()) requireStage(StageName::Compute, "storageBarrier()");
    pushStatement(st::Barrier{BarrierKind::Storage}, loc);
  }, loc);
}

void textureBarrier(Loc loc) {
  return detail::guard([&] {
    if (!buildingDetachedFunction()) requireStage(StageName::Compute, "textureBarrier()");
    pushStatement(st::Barrier{BarrierKind::Texture}, loc);
  }, loc);
}

// ------------------------------------------------------------------ atomics

Val atomicLoad(const Val& target) {
  return detail::guard([&]() -> Val {
    requireAtomicTarget(target, "atomicLoad");
    return Val(makeExpr(ex::Atomic{AtomicOp::Load, target.node(), nullptr},
                        scalarType(target.type()->scalar)));
  }, target);
}

void atomicStore(const Val& target, const ValueLike& value, Loc loc) {
  return detail::guard([&] {
    requireAtomicTarget(target, "atomicStore");
    requireNotVertex("atomicStore");
    const TypePtr& scalar = scalarType(target.type()->scalar);
    const Val v = toVal(value, scalar);
    checkType(v, scalar, "atomicStore");
    pushStatement(st::AtomicStore{target.node(), v.node()}, loc);
  }, target, value, loc);
}

Val atomicAdd(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::Add, t, v, loc); }, t, v, loc);
}
Val atomicSub(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::Sub, t, v, loc); }, t, v, loc);
}
Val atomicMax(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::Max, t, v, loc); }, t, v, loc);
}
Val atomicMin(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::Min, t, v, loc); }, t, v, loc);
}
Val atomicAnd(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::And, t, v, loc); }, t, v, loc);
}
Val atomicOr(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::Or, t, v, loc); }, t, v, loc);
}
Val atomicXor(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val { return atomicOp(AtomicOp::Xor, t, v, loc); }, t, v, loc);
}
Val atomicExchange(const Val& t, const ValueLike& v, Loc loc) {
  return detail::guard([&]() -> Val {
    return atomicOp(AtomicOp::Exchange, t, v, loc);
  }, t, v, loc);
}

// ------------------------------------------------------------------ subgroups

Val subgroupAdd(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupAdd, {v}); }, v);
}
Val subgroupMul(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupMul, {v}); }, v);
}
Val subgroupMin(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupMin, {v}); }, v);
}
Val subgroupMax(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupMax, {v}); }, v);
}
Val subgroupAnd(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupAnd, {v}); }, v);
}
Val subgroupOr(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupOr, {v}); }, v);
}
Val subgroupXor(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupXor, {v}); }, v);
}
Val subgroupAll(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupAll, {v}); }, v);
}
Val subgroupAny(const Val& v) {
  return detail::guard([&]() -> Val { return subgroupCall(BuiltinFunction::SubgroupAny, {v}); }, v);
}

Val subgroupBroadcastFirst(const Val& v) {
  return detail::guard([&]() -> Val {
    return subgroupCall(BuiltinFunction::SubgroupBroadcastFirst, {v});
  }, v);
}

namespace {

/** Converts the id and checks its range when its value is known when the shader is built. */
Val subgroupId(const ValueLike& id, std::string_view what) {
  const Val i = toVal(id, u32());
  requireIndex(i, what);
  if (const auto known = knownRanges(*i.node())) {
    requireSubgroupId(known->front().low, what);
    requireSubgroupId(known->front().high, what);
  }
  return i;
}

} // namespace

Val subgroupBroadcast(const Val& v, const ValueLike& id) {
  return detail::guard([&]() -> Val {
    const Val i = subgroupId(id, "subgroupBroadcast");
    // WGSL needs a const expression; subgroupId() checks the range when it can compute the value
    if (!isConstExpression(*i.node())) throw ShaderError(ErrorCode::ConstantRequired, "subgroupBroadcast: the id must be a constant");
    return subgroupCall(BuiltinFunction::SubgroupBroadcast, {v, i});
  }, v, id);
}

Val subgroupShuffle(const Val& v, const ValueLike& id) {
  return detail::guard([&]() -> Val {
    // a computed id is the shader's to keep in range
    return subgroupCall(BuiltinFunction::SubgroupShuffle, {v, subgroupId(id, "subgroupShuffle")});
  }, v, id);
}

Val subgroupElect() {
  return detail::guard([&]() -> Val {
    return subgroupCall(BuiltinFunction::SubgroupElect, {});
  });
}

Val subgroupBallot(const Val& v) {
  return detail::guard([&]() -> Val {
    return subgroupCall(BuiltinFunction::SubgroupBallot, {v});
  }, v);
}

// ------------------------------------------------------------------ program

ProgramAST computeProgram(std::string name, std::string variantKey, const ComputeOptions& options,
                          const std::function<void()>& body) {
  for (const std::uint32_t n : options.workgroupSize)
    if (n == 0) throw ShaderError(ErrorCode::InvalidDeclaration, std::format("{}: workgroup sizes must be positive", name));
  if (options.subgroupSize && !std::has_single_bit(*options.subgroupSize))
    throw ShaderError(ErrorCode::InvalidDeclaration, std::format("{}: the subgroup size {} is not a power of 2", name, *options.subgroupSize));
  // WebGPU rejects the pipeline otherwise (checked in Chrome 155)
  if (options.subgroupSize && options.workgroupSize[0] % *options.subgroupSize != 0)
    throw ShaderError(ErrorCode::InvalidDeclaration, std::format("{}: the workgroup x size {} is not a multiple of the subgroup size {}", name,
                                  options.workgroupSize[0], *options.subgroupSize));
  const auto prev = currentStage();
  struct Restore {
    std::optional<StageName> prev;
    ~Restore() { currentStage() = prev; }
  } restore{prev};
  ProgramAST ast{std::move(name), std::move(variantKey), {}, {}};
  ast.compute = true;
  ast.workgroupSize = options.workgroupSize;
  ast.subgroupSize = options.subgroupSize;
  currentStage() = StageName::Compute;
  ast.computeBody = collectBlock(body);
  return ast;
}

} // namespace moga
