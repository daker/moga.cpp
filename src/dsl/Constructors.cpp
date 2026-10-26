#include "Context.hpp"
#include "Known.hpp"
#include "moga/dsl/Constructors.hpp"
#include "moga/dsl/Statements.hpp"

#include <algorithm>
#include <format>
#include <optional>

namespace moga {

using namespace detail;

namespace {

/** Converts each argument to its expected type and throws on a mismatch. */
std::vector<ExprPtr> typedArgs(const TypePtr& owner, const std::vector<TypePtr>& types,
                               const std::vector<ValueLike>& args) {
  if (args.size() != types.size())
    throw ShaderError(ErrorCode::ArgumentCount, std::format("{}: expected {} values, got {}", typeName(owner), types.size(),
                                  args.size()));
  std::vector<ExprPtr> nodes;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const Val v = toVal(args[i], types[i]);
    checkType(v, types[i], std::format("{} value {}", typeName(owner), i));
    nodes.push_back(v.node());
  }
  return nodes;
}

Val constructComposite(const TypePtr& type, const std::vector<ValueLike>& args) {
  requireValueType(type, typeName(type));
  std::vector<TypePtr> types;
  // a huge array type with a few arguments must fail here, not after filling a huge list
  const std::size_t expected = type->kind == TypeKind::Struct ? type->fields.size() : static_cast<std::size_t>(type->length);
  if (args.size() != expected)
    throw ShaderError(ErrorCode::ArgumentCount, std::format("{}: expected {} values, got {}", typeName(type), expected, args.size()));
  if (type->kind == TypeKind::Struct)
    for (const StructField& f : type->fields) types.push_back(f.type);
  else
    types.assign(expected, type->element);
  return Val(makeExpr(ex::Construct{typedArgs(type, types, args)}, type));
}

/**
 * A matrix argument copies a matrix of the same type, or makes a mat3 from a
 * mat4, as the only argument. Returns nothing when no argument is a matrix.
 */
std::optional<Val> fromMatrix(const TypePtr& type, const std::vector<Val>& vals) {
  const auto matrix = std::ranges::find_if(vals, [](const Val& v) { return v.type()->kind == TypeKind::Matrix; });
  if (matrix == vals.end()) return std::nullopt;
  const bool single = vals.size() == 1;
  const TypePtr& from = matrix->type();
  // the same shape converts between f32 and f16; a 3x3 takes the upper left of a 4x4 of its kind
  const bool sameShape = single && from->cols == type->cols && from->rows == type->rows;
  const bool upperLeft = single && type->cols == 3 && from->cols == 4 && from->scalar == type->scalar;
  if (!sameShape && !upperLeft)
    throw ShaderError(ErrorCode::TypeMismatch, std::format(
        "{}({}): a matrix must be the only argument and of the same shape (or a 4x4 for a 3x3)",
        typeName(type), typeName(from)));
  // a conversion to f16 can overflow
  Val result(makeExpr(ex::Construct{{matrix->node()}}, type));
  if (const auto reason = nonFiniteConstant(*result.node()))
    throw ShaderError(ErrorCode::InvalidConstant, std::format("{}: {}", typeName(type), *reason));
  return result;
}

/**
 * The matrix forms that GLSL and WGSL share: all components (column major
 * f32 scalars) or one f32 vector per column. One f32 scalar makes a diagonal
 * matrix, as in GLSL; WGSL gets it written out.
 */
Val matrixFromParts(const TypePtr& type, const std::vector<Val>& vals) {
  const auto is = [&](const TypePtr& t) {
    return std::ranges::all_of(vals, [&](const Val& v) { return v.type() == t; });
  };
  const auto count = static_cast<int>(vals.size());
  const TypePtr& scalar = scalarType(type->scalar);
  if (count == 1 && vals[0].type() == scalar) {
    // the diagonal repeats the scalar in WGSL, so a value with side effects runs once
    const Val s = hasSideEffects(*vals[0].node()) ? Let(vals[0], "diagonal") : vals[0];
    return Val(makeExpr(ex::Construct{{s.node()}}, type));
  }
  const bool components = is(scalar) && count == type->cols * type->rows;
  const bool columns = is(vectorType(type->scalar, type->rows)) && count == type->cols;
  if (!components && !columns) {
    std::string list;
    for (const Val& v : vals) list += (list.empty() ? "" : ", ") + typeName(v.type());
    throw ShaderError(ErrorCode::ArgumentCount, std::format("{}({}): give one {} (a diagonal), {} {} components or {} {} columns",
                                  typeName(type), list, typeName(scalar), type->cols * type->rows,
                                  typeName(scalar), type->cols, typeName(vectorType(type->scalar, type->rows))));
  }
  std::vector<ExprPtr> nodes;
  for (const Val& v : vals) nodes.push_back(v.node());
  return Val(makeExpr(ex::Construct{std::move(nodes)}, type));
}

/**
 * WGSL converts the kind of one value of the same size; otherwise (several
 * parts, or one scalar filling a vector) every part must have the target kind.
 */
void requirePartKinds(const TypePtr& type, const std::vector<Val>& vals) {
  const ScalarKind want = scalarOf(type);
  for (const Val& v : vals)
    if (scalarOf(v.type()) != want)
      throw ShaderError(ErrorCode::TypeMismatch, std::format("{}: part {} has {} components; convert it first", typeName(type),
                                    typeName(v.type()), typeName(scalarType(scalarOf(v.type())))));
}

} // namespace

Val construct(const TypePtr& type, const std::vector<ValueLike>& args) {
  return detail::guard([&]() -> Val {
    if (type->kind == TypeKind::Struct || type->kind == TypeKind::Array)
      return constructComposite(type, args);
    const ScalarKind scalar = scalarOf(type);
    std::vector<Val> vals;
    for (const ValueLike& a : args) vals.push_back(toVal(a, scalarType(scalar)));
    if (const std::optional<Val> m = fromMatrix(type, vals)) return *m;
    if (type->kind == TypeKind::Matrix) return matrixFromParts(type, vals);
    const int want = componentCount(type);
    const bool single = vals.size() == 1;
    int have = 0;
    for (const Val& v : vals) have += componentCount(v.type());
    const bool splat = single && vals[0].type()->kind == TypeKind::Scalar;
    const bool convert = single && componentCount(vals[0].type()) == want;
    if (!convert || (splat && type->kind == TypeKind::Vector)) requirePartKinds(type, vals);
    if (!splat && !convert && have != want) {
      std::string list;
      for (const Val& v : vals) list += (list.empty() ? "" : ", ") + typeName(v.type());
      throw ShaderError(ErrorCode::ArgumentCount, std::format("{}({}): component count mismatch", typeName(type), list));
    }
    std::vector<ExprPtr> nodes;
    for (const Val& v : vals) nodes.push_back(v.node());
    // a conversion to f16 can overflow
    Val result(makeExpr(ex::Construct{std::move(nodes)}, type));
    if (const auto reason = nonFiniteConstant(*result.node()))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("{}: {}", typeName(type), *reason));
    return result;
  }, type, args);
}

Val array(const std::vector<ValueLike>& elements) {
  return detail::guard([&]() -> Val {
    if (elements.empty()) throw ShaderError(ErrorCode::ArgumentCount, "array() needs at least one element");
    const Val first = toVal(elements.front());
    return constructComposite(arrayOf(first.type(), static_cast<int>(elements.size())), elements);
  }, elements);
}

Val toF32(const ValueLike& a) {
  return detail::guard([&]() -> Val { return a.number() ? toVal(a, f32()) : construct(f32(), {a}); }, a);
}
Val toI32(const ValueLike& a) {
  return detail::guard([&]() -> Val { return a.number() ? toVal(a, i32()) : construct(i32(), {a}); }, a);
}
Val toF16(const ValueLike& a) {
  return detail::guard([&]() -> Val { return a.number() ? toVal(a, f16()) : construct(f16(), {a}); }, a);
}
Val toU32(const ValueLike& a) {
  return detail::guard([&]() -> Val { return a.number() ? toVal(a, u32()) : construct(u32(), {a}); }, a);
}
Val toBool(const ValueLike& a) {
  return detail::guard([&]() -> Val { return a.val() ? construct(boolType(), {a}) : toVal(a, boolType()); }, a);
}

Val cast(const Val& v, ScalarKind scalar) {
  return detail::guard([&]() -> Val {
    const TypePtr& t = v.type();
    if (!isScalarOrVector(t)) throw ShaderError(ErrorCode::TypeMismatch, std::format("Cannot cast {}", typeName(t)));
    return construct(vectorType(scalar, t->kind == TypeKind::Vector ? t->size : 1), {v});
  }, v, scalar);
}

Val convert(const Val& v, const TypePtr& type) {
  return detail::guard([&]() -> Val {
    const TypePtr& from = v.type();
    if (!isScalarOrVector(from) || !isScalarOrVector(type))
      throw ShaderError(ErrorCode::TypeMismatch, std::format("Cannot convert {} to {}", typeName(from), typeName(type)));
    if (typesEqual(from, type)) return v;
    const int have = componentCount(from);
    const int want = componentCount(type);
    // a scalar fills the vector after it gets the target kind
    if (have == 1) return construct(type, {from->scalar == type->scalar ? v : cast(v, type->scalar)});
    if (have == want) return construct(type, {v});
    if (have > want) return construct(type, {v.swizzle(std::string("xyzw").substr(0, want))});
    std::vector<ValueLike> parts{from->scalar == type->scalar ? v : cast(v, type->scalar)};
    for (int i = have; i < want; ++i) parts.emplace_back(0);
    return construct(type, parts);
  }, v, type);
}

} // namespace moga
