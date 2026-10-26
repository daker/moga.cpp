#include "BuiltinTable.hpp"

#include <array>
#include <format>

namespace moga::detail {

namespace {

using B = BuiltinFunction;
using D = Domain;
using R = Result;

constexpr BuiltinSpec spec(B fn, std::string_view name, std::string_view glsl, std::string_view wgsl,
                           D domain, R result) {
  return {fn, name, glsl, wgsl, domain, result};
}

/** Elementwise float function with the same name in the DSL and both languages. */
constexpr BuiltinSpec floatFn(B fn, std::string_view name) {
  return spec(fn, name, name, name, D::FloatScalarOrVector, R::SameAsFirst);
}

constexpr BuiltinSpec fragmentOnly(BuiltinSpec s) {
  s.stages = Stages::Fragment;
  s.constEval = false;
  return s;
}

constexpr BuiltinSpec runtimeOnly(BuiltinSpec s) {
  s.constEval = false;
  return s;
}

constexpr BuiltinSpec subgroupFn(B fn, std::string_view name, D domain, R result) {
  BuiltinSpec s = spec(fn, name, name, name, domain, result);
  s.stages = Stages::ComputeOrFragment;
  s.constEval = false;
  s.subgroup = true;
  return s;
}

constexpr BuiltinSpec bitcastFn(B fn, std::string_view name, D domain, R result) {
  BuiltinSpec s = spec(fn, name, name, "bitcast", domain, result);
  s.bitcast = true;
  return s;
}

// in the order of the BuiltinFunction enum (checked below)
constexpr std::array kTable = {
    spec(B::Abs, "abs", "abs", "abs", D::Numeric, R::SameAsFirst),
    // WGSL and GLSL ES 3.00 both have sign() for i32 too
    spec(B::Sign, "sign", "sign", "sign", D::SignedNumeric, R::SameAsFirst),
    floatFn(B::Floor, "floor"),
    floatFn(B::Ceil, "ceil"),
    floatFn(B::Fract, "fract"),
    // GLSL round() may go either way at .5; WGSL round() rounds half to even
    spec(B::Round, "round", "roundEven", "round", D::FloatScalarOrVector, R::SameAsFirst),
    floatFn(B::Trunc, "trunc"),
    floatFn(B::Sqrt, "sqrt"),
    spec(B::InverseSqrt, "inverseSqrt", "inversesqrt", "inverseSqrt", D::FloatScalarOrVector, R::SameAsFirst),
    floatFn(B::Exp, "exp"),
    floatFn(B::Exp2, "exp2"),
    floatFn(B::Log, "log"),
    floatFn(B::Log2, "log2"),
    floatFn(B::Pow, "pow"),
    floatFn(B::Sin, "sin"),
    floatFn(B::Cos, "cos"),
    floatFn(B::Tan, "tan"),
    floatFn(B::Asin, "asin"),
    floatFn(B::Acos, "acos"),
    floatFn(B::Atan, "atan"),
    spec(B::Atan2, "atan2", "atan", "atan2", D::FloatScalarOrVector, R::SameAsFirst),
    spec(B::Min, "min", "min", "min", D::Numeric, R::SameAsFirst),
    spec(B::Max, "max", "max", "max", D::Numeric, R::SameAsFirst),
    spec(B::Clamp, "clamp", "clamp", "clamp", D::Numeric, R::SameAsFirst),
    floatFn(B::Saturate, "saturate"),
    floatFn(B::Mix, "mix"),
    floatFn(B::Step, "step"),
    floatFn(B::Smoothstep, "smoothstep"),
    spec(B::Length, "length", "length", "length", D::FloatScalarOrVector, R::ScalarOfFirst),
    spec(B::Distance, "distance", "distance", "distance", D::FloatScalarOrVector, R::ScalarOfFirst),
    spec(B::Dot, "dot", "dot", "dot", D::NumericVector, R::ScalarOfFirst),
    spec(B::Cross, "cross", "cross", "cross", D::Vec3f, R::SameAsFirst),
    spec(B::Normalize, "normalize", "normalize", "normalize", D::FloatVector, R::SameAsFirst),
    spec(B::Reflect, "reflect", "reflect", "reflect", D::FloatVector, R::SameAsFirst),
    spec(B::FaceForward, "faceForward", "faceforward", "faceForward", D::FloatVector, R::SameAsFirst),
    floatFn(B::Mod, "mod"),
    fragmentOnly(spec(B::Dpdx, "dpdx", "dFdx", "dpdx", D::F32ScalarOrVector, R::SameAsFirst)),
    fragmentOnly(spec(B::Dpdy, "dpdy", "dFdy", "dpdy", D::F32ScalarOrVector, R::SameAsFirst)),
    fragmentOnly(spec(B::Fwidth, "fwidth", "fwidth", "fwidth", D::F32ScalarOrVector, R::SameAsFirst)),
    spec(B::Transpose, "transpose", "transpose", "transpose", D::Matrix, R::SameAsFirst),
    spec(B::Determinant, "determinant", "determinant", "determinant", D::SquareMatrix, R::ScalarOfFirst),
    spec(B::Any, "any", "any", "any", D::BoolScalarOrVector, R::Bool),
    spec(B::All, "all", "all", "all", D::BoolScalarOrVector, R::Bool),
    bitcastFn(B::FloatBitsToUint, "floatBitsToUint", D::F32Bits, R::ShapeU32),
    bitcastFn(B::UintBitsToFloat, "uintBitsToFloat", D::U32Bits, R::ShapeF32),
    bitcastFn(B::FloatBitsToInt, "floatBitsToInt", D::F32Bits, R::ShapeI32),
    bitcastFn(B::IntBitsToFloat, "intBitsToFloat", D::I32Bits, R::ShapeF32),
    floatFn(B::Sinh, "sinh"),
    floatFn(B::Cosh, "cosh"),
    floatFn(B::Tanh, "tanh"),
    floatFn(B::Asinh, "asinh"),
    floatFn(B::Acosh, "acosh"),
    floatFn(B::Atanh, "atanh"),
    floatFn(B::Radians, "radians"),
    floatFn(B::Degrees, "degrees"),
    spec(B::Refract, "refract", "refract", "refract", D::FloatVector, R::SameAsFirst),
    // WGSL gets inverse from a generated helper function, which cannot run at compile time
    runtimeOnly(spec(B::Inverse, "inverse", "inverse", "inverse", D::SquareMatrix, R::SameAsFirst)),
    spec(B::PackHalf2x16, "packHalf2x16", "packHalf2x16", "pack2x16float", D::Vec2f, R::U32),
    spec(B::PackSnorm2x16, "packSnorm2x16", "packSnorm2x16", "pack2x16snorm", D::Vec2f, R::U32),
    spec(B::PackUnorm2x16, "packUnorm2x16", "packUnorm2x16", "pack2x16unorm", D::Vec2f, R::U32),
    spec(B::UnpackHalf2x16, "unpackHalf2x16", "unpackHalf2x16", "unpack2x16float", D::U32, R::Vec2f),
    spec(B::UnpackSnorm2x16, "unpackSnorm2x16", "unpackSnorm2x16", "unpack2x16snorm", D::U32, R::Vec2f),
    spec(B::UnpackUnorm2x16, "unpackUnorm2x16", "unpackUnorm2x16", "unpack2x16unorm", D::U32, R::Vec2f),
    subgroupFn(B::SubgroupAdd, "subgroupAdd", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupMul, "subgroupMul", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupMin, "subgroupMin", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupMax, "subgroupMax", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupAnd, "subgroupAnd", D::Integer, R::SameAsFirst),
    subgroupFn(B::SubgroupOr, "subgroupOr", D::Integer, R::SameAsFirst),
    subgroupFn(B::SubgroupXor, "subgroupXor", D::Integer, R::SameAsFirst),
    subgroupFn(B::SubgroupAll, "subgroupAll", D::Bool, R::Bool),
    subgroupFn(B::SubgroupAny, "subgroupAny", D::Bool, R::Bool),
    subgroupFn(B::SubgroupBroadcastFirst, "subgroupBroadcastFirst", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupBroadcast, "subgroupBroadcast", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupShuffle, "subgroupShuffle", D::Numeric, R::SameAsFirst),
    subgroupFn(B::SubgroupElect, "subgroupElect", D::None, R::Bool),
    subgroupFn(B::SubgroupBallot, "subgroupBallot", D::Bool, R::Vec4u),
};

constexpr bool inEnumOrder() {
  for (std::size_t i = 0; i < kTable.size(); ++i)
    if (static_cast<std::size_t>(kTable[i].fn) != i) return false;
  return kTable.size() == static_cast<std::size_t>(B::SubgroupBallot) + 1;
}
static_assert(inEnumOrder(), "kTable rows must follow the BuiltinFunction enum");

bool scalarOrVectorOf(const TypePtr& t, ScalarKind kind) {
  return (t->kind == TypeKind::Scalar || t->kind == TypeKind::Vector) && t->scalar == kind;
}

int sizeOf(const TypePtr& t) { return t->kind == TypeKind::Vector ? t->size : 1; }

} // namespace

const BuiltinSpec& builtinSpec(BuiltinFunction fn) { return kTable[static_cast<std::size_t>(fn)]; }

namespace {

bool scalarOrVector(const TypePtr& t) { return t->kind == TypeKind::Scalar || t->kind == TypeKind::Vector; }

struct DomainRow {
  bool (*test)(const TypePtr&);
  std::string_view text;
};

// in the order of the Domain enum (checked below)
constexpr std::array<DomainRow, 18> kDomains = {{
    {[](const TypePtr&) { return false; }, "no argument"},
    {[](const TypePtr& t) { return scalarOrVector(t) && isFloatKind(t->scalar); }, "a float scalar or vector"},
    {[](const TypePtr& t) { return scalarOrVectorOf(t, ScalarKind::F32); }, "an f32 scalar or vector"},
    {[](const TypePtr& t) { return t->kind == TypeKind::Vector && isFloatKind(t->scalar); }, "a float vector"},
    {[](const TypePtr& t) { return t == vec3f(); }, "a vec3f"},
    {[](const TypePtr& t) { return t == vec2f(); }, "a vec2f"},
    {[](const TypePtr& t) { return t == u32(); }, "a u32"},
    {[](const TypePtr& t) { return isNumeric(t); }, "a numeric scalar or vector"},
    {[](const TypePtr& t) { return t->kind == TypeKind::Vector && isNumeric(t); }, "a numeric vector"},
    {[](const TypePtr& t) { return isInteger(t); }, "an integer scalar or vector"},
    {[](const TypePtr& t) { return t == boolType(); }, "a bool"},
    {[](const TypePtr& t) { return isBool(t); }, "a bool or a bool vector"},
    {[](const TypePtr& t) { return t->kind == TypeKind::Matrix; }, "a matrix"},
    {[](const TypePtr& t) { return t->kind == TypeKind::Matrix && t->cols == t->rows; }, "a square matrix"},
    {[](const TypePtr& t) { return scalarOrVectorOf(t, ScalarKind::F32); }, "an f32 scalar or vector"},
    {[](const TypePtr& t) { return scalarOrVectorOf(t, ScalarKind::U32); }, "a u32 scalar or vector"},
    {[](const TypePtr& t) { return scalarOrVectorOf(t, ScalarKind::I32); }, "an i32 scalar or vector"},
    {[](const TypePtr& t) { return scalarOrVector(t) && (isFloatKind(t->scalar) || t->scalar == ScalarKind::I32); },
     "a float or i32 scalar or vector"},
}};
static_assert(kDomains.size() == static_cast<std::size_t>(D::SignedNumeric) + 1, "kDomains rows must follow the Domain enum");

} // namespace

bool inDomain(Domain domain, const TypePtr& t) { return kDomains[static_cast<std::size_t>(domain)].test(t); }

std::string_view domainText(Domain domain) { return kDomains[static_cast<std::size_t>(domain)].text; }

namespace {

TypePtr resultOf(Result result, const TypePtr& first) {
  switch (result) {
  case R::SameAsFirst:
    return first;
  case R::F32:
    return f32();
  case R::U32:
    return u32();
  case R::Vec2f:
    return vec2f();
  case R::Vec4u:
    return vec4u();
  case R::Bool:
    return boolType();
  case R::ScalarOfFirst:
    return scalarType(first->scalar);
  case R::ShapeU32:
    return vectorType(ScalarKind::U32, sizeOf(first));
  case R::ShapeF32:
    return vectorType(ScalarKind::F32, sizeOf(first));
  case R::ShapeI32:
    return vectorType(ScalarKind::I32, sizeOf(first));
  }
  return first;
}

} // namespace

TypePtr checkBuiltin(BuiltinFunction fn, const TypePtr& first) {
  const BuiltinSpec& s = builtinSpec(fn);
  if (s.domain == D::None) return resultOf(s.result, first);
  if (!inDomain(s.domain, first))
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{} needs {}, got {}", s.name, domainText(s.domain), typeName(first)));
  return resultOf(s.result, first);
}

} // namespace moga::detail
