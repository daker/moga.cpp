#include "Context.hpp"
#include "Known.hpp"
#include "moga/dsl/Value.hpp"
#include "moga/dsl/Constructors.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace moga {

using namespace detail;

namespace {

using OperandRule = TypePtr (*)(const TypePtr&, const TypePtr&);

TypePtr logicalType(const TypePtr& a, const TypePtr& b) {
  return a == boolType() && b == boolType() ? boolType() : nullptr;
}

/** == and != compare any scalars or vectors of one type, bools included. */
TypePtr equalityType(const TypePtr& a, const TypePtr& b) {
  if (!typesEqual(a, b)) return nullptr;
  if (a->kind == TypeKind::Vector) return vectorType(ScalarKind::Bool, a->size);
  return a->kind == TypeKind::Scalar ? boolType() : nullptr;
}

/** <, <=, > and >= need numeric operands. */
TypePtr orderedType(const TypePtr& a, const TypePtr& b) {
  return isNumeric(a) ? equalityType(a, b) : nullptr;
}

/** &, | and ^ need two integers of one type (binary() widens a scalar first). */
TypePtr bitwiseType(const TypePtr& a, const TypePtr& b) {
  return isInteger(a) && typesEqual(a, b) ? a : nullptr;
}

/**
 * A shift keeps the type of the shifted value. The amount has the same shape:
 * a vector needs a vector amount of its size (binary() widens a scalar first).
 */
TypePtr shiftType(const TypePtr& a, const TypePtr& b) {
  if (!isInteger(a) || !isInteger(b)) return nullptr;
  const int aSize = a->kind == TypeKind::Vector ? a->size : 1;
  const int bSize = b->kind == TypeKind::Vector ? b->size : 1;
  return aSize == bSize ? a : nullptr;
}

/** Componentwise arithmetic on one scalar kind; a scalar operand broadcasts. */
TypePtr componentwiseType(const TypePtr& a, const TypePtr& b) {
  if (!isNumeric(a) || !isNumeric(b) || scalarOf(a) != scalarOf(b)) return nullptr;
  if (a->kind == TypeKind::Vector && b->kind == TypeKind::Vector)
    return a->size == b->size ? a : nullptr;
  return a->kind == TypeKind::Vector ? a : b;
}

/** % is integer only here (mod() is the float form), with the shapes of componentwise arithmetic. */
TypePtr remainderType(const TypePtr& a, const TypePtr& b) {
  return isInteger(a) && isInteger(b) ? componentwiseType(a, b) : nullptr;
}

bool hasMatrix(const TypePtr& a, const TypePtr& b) {
  return a->kind == TypeKind::Matrix || b->kind == TypeKind::Matrix;
}

/** `m * b`: matrix product, matrix times column vector, or scaling by f32. */
TypePtr matrixTimes(const TypePtr& m, const TypePtr& b) {
  if (b->kind == TypeKind::Matrix)
    return m->cols == b->rows && m->scalar == b->scalar ? squareMatrix(m->rows, m->scalar) : nullptr;
  if (b->kind == TypeKind::Vector)
    return b->scalar == m->scalar && b->size == m->cols ? vectorType(m->scalar, m->rows) : nullptr;
  return b == scalarType(m->scalar) ? m : nullptr;
}

/** `a * m` with a non matrix left operand: row vector times matrix, or f32 scaling. */
TypePtr timesMatrix(const TypePtr& a, const TypePtr& m) {
  if (a->kind == TypeKind::Vector)
    return a->scalar == m->scalar && a->size == m->rows ? vectorType(m->scalar, m->cols) : nullptr;
  return a == scalarType(m->scalar) ? m : nullptr;
}

TypePtr productType(const TypePtr& a, const TypePtr& b) {
  if (a->kind == TypeKind::Matrix) return matrixTimes(a, b);
  if (b->kind == TypeKind::Matrix) return timesMatrix(a, b);
  return componentwiseType(a, b);
}

/** Matrices add and subtract only with an identical matrix type. */
TypePtr sumType(const TypePtr& a, const TypePtr& b) {
  if (!hasMatrix(a, b)) return componentwiseType(a, b);
  return typesEqual(a, b) ? a : nullptr;
}

TypePtr quotientType(const TypePtr& a, const TypePtr& b) {
  return hasMatrix(a, b) ? nullptr : componentwiseType(a, b);
}

OperandRule ruleOf(BinaryOp op) {
  switch (op) {
  case BinaryOp::And:
  case BinaryOp::Or:
    return logicalType;
  case BinaryOp::Lt:
  case BinaryOp::Le:
  case BinaryOp::Gt:
  case BinaryOp::Ge:
    return orderedType;
  case BinaryOp::Eq:
  case BinaryOp::Ne:
    return equalityType;
  case BinaryOp::BitAnd:
  case BinaryOp::BitOr:
  case BinaryOp::BitXor:
    return bitwiseType;
  case BinaryOp::Shl:
  case BinaryOp::Shr:
    return shiftType;
  case BinaryOp::Rem:
    return remainderType;
  case BinaryOp::Add:
  case BinaryOp::Sub:
    return sumType;
  case BinaryOp::Mul:
    return productType;
  case BinaryOp::Div:
    return quotientType;
  }
  return nullptr;
}

std::vector<int> swizzleIndices(std::string_view comps) {
  for (std::string_view set : {std::string_view("xyzw"), std::string_view("rgba")}) {
    std::vector<int> idx;
    for (char c : comps) {
      const auto p = set.find(c);
      if (p == std::string_view::npos) break;
      idx.push_back(static_cast<int>(p));
    }
    if (idx.size() == comps.size() && !idx.empty()) return idx;
  }
  throw ShaderError(ErrorCode::UnknownName, std::format("Invalid swizzle \"{}\"", comps));
}

void assertAssignable(const Expr& e) {
  if (const auto* l = std::get_if<ex::Local>(&e.node)) {
    if (!l->v->mutableVar)
      throw ShaderError(ErrorCode::NotAssignable, std::format("Cannot assign to immutable \"{}\"", l->v->name));
    return;
  }
  if (const auto* s = std::get_if<ex::Swizzle>(&e.node)) {
    // a scalar swizzle of more than one component is a new vector, and no
    // backend writes the same component twice
    const bool scalarBase = s->base->type->kind == TypeKind::Scalar;
    const bool repeats = std::ranges::any_of(s->comps, [&](char c) {
      return std::ranges::count(s->comps, c) > 1;
    });
    if ((scalarBase && s->comps.size() > 1) || repeats)
      throw ShaderError(ErrorCode::NotAssignable, std::format("Cannot assign to swizzle \"{}\"", s->comps));
    return assertAssignable(*s->base);
  }
  if (const auto* m = std::get_if<ex::Member>(&e.node)) return assertAssignable(*m->base);
  if (const auto* i = std::get_if<ex::Index>(&e.node)) return assertAssignable(*i->base);
  if (const auto* b = std::get_if<ex::StorageElement>(&e.node)) {
    if (b->buffer->access != StorageAccess::ReadWrite)
      throw ShaderError(ErrorCode::NotAssignable, std::format("Cannot assign to read only storage \"{}\"", b->buffer->name));
    if (currentStage() == StageName::Vertex)
      throw ShaderError(ErrorCode::WrongStage, "Storage writes are not valid in the vertex stage");
    return;
  }
  if (std::holds_alternative<ex::Workgroup>(e.node)) return;
  throw ShaderError(ErrorCode::NotAssignable, "Expression is not assignable");
}

const TypePtr& operandLike(const Val& v) {
  return v.type()->kind == TypeKind::Matrix ? scalarType(v.type()->scalar) : v.type();
}

} // namespace

// ------------------------------------------------------------------ values

Val toVal(const ValueLike& v, const TypePtr& like) {
  return detail::guard([&]() -> Val {
    if (const Val* x = v.val()) return *x;
    if (const bool* b = v.boolean()) return literal(*b, boolType());
    const double n = *v.number();
    ScalarKind kind = ScalarKind::F32;
    if (like && (isScalarOrVector(like) || like->kind == TypeKind::Matrix)) kind = scalarOf(like);
    if (kind == ScalarKind::Bool) return literal(n != 0, boolType());
    if (!isFloatKind(kind) && n != std::trunc(n))
      throw ShaderError(ErrorCode::TypeMismatch, std::format("Non-integer literal {} used as {}", n, typeName(scalarType(kind))), v.where());
    // the backends cast integers through int64; check the range before that
    const auto outside = [&](double lo, double hi) { return !(n >= lo && n <= hi); };
    if (kind == ScalarKind::I32 && outside(kI32Min, kI32Max))
      throw ShaderError(ErrorCode::OutOfRange, std::format("Literal {} does not fit in i32", n), v.where());
    if (kind == ScalarKind::U32 && outside(0.0, kU32Max))
      throw ShaderError(ErrorCode::OutOfRange, std::format("Literal {} does not fit in u32", n), v.where());
    if (kind == ScalarKind::F32 && outside(-kF32Max, kF32Max))
      throw ShaderError(ErrorCode::OutOfRange, std::format("Literal {} does not fit in f32", n), v.where());
    if (kind == ScalarKind::F16 && outside(-kF16Max, kF16Max))
      throw ShaderError(ErrorCode::OutOfRange, std::format("Literal {} does not fit in f16", n), v.where());
    return literal(n, scalarType(kind));
  }, v, like);
}

namespace {

/**
 * WGSL rejects an integer / or % whose divisor is a const expression with a
 * zero component, so a divisor that is known when the shader is built is
 * checked now. Float division by zero is valid.
 */
void requireNonZeroDivisor(BinaryOp op, const Val& divisor, const Loc& at) {
  if ((op != BinaryOp::Div && op != BinaryOp::Rem) || !isInteger(divisor.type())) return;
  const auto known = detail::knownRanges(*divisor.node());
  if (known && std::ranges::any_of(*known, [](const detail::KnownRange& r) { return detail::mayBe(r, 0); }))
    throw ShaderError(ErrorCode::InvalidConstant, std::format("Integer \"{}\" by a constant zero", opText(op)), at);
}

/** WGSL rejects the one i32 quotient that overflows, -2147483648 / -1 (and % -1), when both are known. */
void requireNoQuotientOverflow(BinaryOp op, const Val& dividend, const Val& divisor, const Loc& at) {
  if ((op != BinaryOp::Div && op != BinaryOp::Rem) || dividend.type()->scalar != ScalarKind::I32) return;
  const auto a = detail::knownRanges(*dividend.node());
  const auto b = a ? detail::knownRanges(*divisor.node()) : std::nullopt;
  if (!b) return;
  for (std::size_t i = 0; i < std::max(a->size(), b->size()); ++i)
    if (detail::mayBe((*a)[a->size() == 1 ? 0 : i], detail::kI32Min) && detail::mayBe((*b)[b->size() == 1 ? 0 : i], -1))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("Integer \"{}\" of -2147483648 by -1 overflows i32", opText(op)), at);
}

/**
 * WGSL rejects a const expression shift outside 0..31, and a left shift whose
 * result does not fit the type: a 1 bit shifted out of a u32, or an i32
 * result outside the i32 range (Tint, Chrome 155: 1i << 31u and 3u << 31u
 * fail, -1i << 31u and 1u << 31u do not).
 */
void requireValidShift(BinaryOp op, const Val& value, const Val& amount, const Loc& valueAt, const Loc& amountAt) {
  if (op != BinaryOp::Shl && op != BinaryOp::Shr) return;
  const auto amounts = detail::knownRanges(*amount.node());
  if (!amounts) return;
  // the emitter converts an i32 amount to u32, so a negative one becomes a huge shift
  for (const detail::KnownRange& s : *amounts)
    if (s.low < 0 || s.high >= 32)
      throw ShaderError(ErrorCode::OutOfRange, std::format("Shift by {} is not in 0..31", detail::describe(s)), amountAt);
  const auto values = detail::knownRanges(*value.node());
  if (op != BinaryOp::Shl || !values) return;
  const bool isUnsigned = value.type()->scalar == ScalarKind::U32;
  // the largest shift of the value of largest magnitude, at each end, is the one that can overflow
  for (std::size_t i = 0; i < values->size(); ++i) {
    const double s = (*amounts)[amounts->size() == 1 ? 0 : i].high;
    for (const double v : {(*values)[i].low, (*values)[i].high}) {
      const double shifted = v * std::ldexp(1.0, static_cast<int>(s));
      const bool fits = isUnsigned ? shifted <= detail::kU32Max : shifted >= detail::kI32Min && shifted <= detail::kI32Max;
      if (!fits)
        throw ShaderError(ErrorCode::OutOfRange, std::format("{} << {} does not fit in {}", v, s, typeName(scalarType(value.type()->scalar))), valueAt);
    }
  }
}

} // namespace

Val binary(BinaryOp op, const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val {
    // a literal takes the type of a shader value operand; two literals are f32
    const Val lhs = a.val() ? *a.val() : toVal(a, b.val() ? operandLike(*b.val()) : f32());
    Val rhs = toVal(b, operandLike(lhs));
    Val left = lhs;
    // WGSL bitwise operators need one type and shifts a vector amount for a
    // vector: widen a scalar of the same kind to the vector
    const bool bitwise = op == BinaryOp::BitAnd || op == BinaryOp::BitOr || op == BinaryOp::BitXor;
    const bool shift = op == BinaryOp::Shl || op == BinaryOp::Shr;
    const auto isVec = [](const Val& v) { return v.type()->kind == TypeKind::Vector; };
    const auto isInt = [](const Val& v) { return isInteger(v.type()); };
    if ((bitwise || shift) && isInt(left) && isInt(rhs)) {
      if (isVec(left) && !isVec(rhs) && (shift || rhs.type()->scalar == left.type()->scalar))
        rhs = construct(vectorType(rhs.type()->scalar, left.type()->size), {rhs});
      else if (bitwise && !isVec(left) && isVec(rhs) && left.type()->scalar == rhs.type()->scalar)
        left = construct(rhs.type(), {left});
    }
    const TypePtr type = ruleOf(op)(left.type(), rhs.type());
    if (!type)
      throw ShaderError(ErrorCode::InvalidOperands, std::format("Invalid operands for \"{}\": {} and {}", opText(op),
                                    typeName(left.type()), typeName(rhs.type())), b.where());
    // each error names the operand at fault: the right one, or the shifted value
    requireNonZeroDivisor(op, rhs, b.where());
    requireNoQuotientOverflow(op, left, rhs, b.where());
    requireValidShift(op, left, rhs, a.where(), b.where());
    Val result(makeExpr(ex::Binary{op, left.node(), rhs.node()}, type));
    if (const auto reason = detail::nonFiniteConstant(*result.node()))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("\"{}\": {}", opText(op), *reason), b.where());
    return result;
  }, op, a, b);
}

Val unary(UnaryOp op, const Val& a, const Loc& at) {
  return detail::guard([&]() -> Val {
    const TypePtr& t = a.type();
    if (op == UnaryOp::Not && t != boolType())
      throw ShaderError(ErrorCode::InvalidOperands, std::format("\"!\" needs bool, got {}", typeName(t)), at);
    // WGSL has no unary minus for u32
    const bool signedNumber = isNumeric(t) && t->scalar != ScalarKind::U32;
    if (op == UnaryOp::Neg && !signedNumber && t->kind != TypeKind::Matrix)
      throw ShaderError(ErrorCode::InvalidOperands, std::format("\"-\" needs f32, i32 or a matrix, got {}", typeName(t)), at);
    if (op == UnaryOp::BitNot && !isInteger(t))
      throw ShaderError(ErrorCode::InvalidOperands, std::format("\"~\" needs integer, got {}", typeName(t)), at);
    return Val(makeExpr(ex::Unary{op, a.node()}, t));
  }, op, a, at);
}

Val Val::swizzle(std::string_view comps) const {
  return detail::guard([&]() -> Val {
    const TypePtr& t = type();
    if (!isScalarOrVector(t)) throw ShaderError(ErrorCode::InvalidOperands, std::format("Cannot swizzle {}", typeName(t)));
    const int size = t->kind == TypeKind::Vector ? t->size : 1;
    const std::vector<int> idx = swizzleIndices(comps);
    if (idx.size() > 4 || std::ranges::any_of(idx, [&](int i) { return i >= size; }))
      throw ShaderError(ErrorCode::OutOfRange, std::format("Swizzle \"{}\" out of range for {}", comps, typeName(t)));
    const TypePtr& result = vectorType(t->scalar, static_cast<int>(idx.size()));
    // a swizzle of a swizzle is one swizzle of the base: v.xy.yx is v.yx, which
    // a backend can also write
    if (const auto* inner = std::get_if<ex::Swizzle>(&node_->node)) {
      std::string composed;
      for (int i : idx) composed += inner->comps[static_cast<std::size_t>(i)];
      return Val(makeExpr(ex::Swizzle{inner->base, composed}, result));
    }
    std::string canonical;
    for (int i : idx) canonical += "xyzw"[i];
    return Val(makeExpr(ex::Swizzle{node_, canonical}, result));
  }, *this, comps);
}

Val Val::member(std::string_view field) const {
  return detail::guard([&]() -> Val {
    const TypePtr& t = type();
    if (t->kind != TypeKind::Struct)
      throw ShaderError(ErrorCode::InvalidOperands, std::format("Member \"{}\" on non-struct {}", field, typeName(t)));
    for (const StructField& f : t->fields)
      if (f.name == field) return Val(makeExpr(ex::Member{node_, f.name}, f.type));
    throw ShaderError(ErrorCode::UnknownName, std::format("Struct {} has no field \"{}\"", t->name, field));
  }, *this, field);
}

Val Val::operator[](const ValueLike& index) const {
  return detail::guard([&]() -> Val {
    const TypePtr& t = type();
    const Val i = toVal(index, i32());
    if (!isInteger(i.type()) || i.type()->kind != TypeKind::Scalar)
      throw ShaderError(ErrorCode::TypeMismatch, "Index must be an integer scalar");
    TypePtr result;
    int count = 0;
    if (t->kind == TypeKind::Array) {
      result = t->element;
      count = t->length;
    } else if (t->kind == TypeKind::Matrix) {
      result = vectorType(t->scalar, t->rows);
      count = t->cols;
    } else if (t->kind == TypeKind::Vector) {
      result = scalarType(t->scalar);
      count = t->size;
    } else {
      throw ShaderError(ErrorCode::InvalidOperands, std::format("Cannot index {}", typeName(t)));
    }
    // WGSL rejects a const expression index out of range; a run time index is the shader's to keep in range
    if (const auto known = detail::knownRanges(*i.node()); known && (known->front().low < 0 || known->front().high >= count))
      throw ShaderError(ErrorCode::OutOfRange, 
          std::format("Index {} is out of range for {} (0..{})", detail::describe(known->front()), typeName(t), count - 1));
    return Val(makeExpr(ex::Index{node_, i.node()}, result));
  }, *this, index);
}

Val Val::eq(const ValueLike& o) const {
  return detail::guard([&]() -> Val { return binary(BinaryOp::Eq, *this, o); }, *this, o);
}
Val Val::ne(const ValueLike& o) const {
  return detail::guard([&]() -> Val { return binary(BinaryOp::Ne, *this, o); }, *this, o);
}

void Val::set(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    assertAssignable(*node_);
    if (containsAtomic(type())) throw ShaderError(ErrorCode::NotAssignable, "Use atomicStore() to write an atomic");
    const Val v = toVal(value, type());
    if (!typesEqual(v.type(), type()))
      throw ShaderError(ErrorCode::TypeMismatch, 
          std::format("Cannot assign {} to {}", typeName(v.type()), typeName(type())));
    pushStatement(st::Assign{node_, v.node()}, loc);
  }, *this, value, loc);
}

void Val::addAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Add, value, loc);
  }, *this, value, loc);
}
void Val::subAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Sub, value, loc);
  }, *this, value, loc);
}
void Val::mulAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Mul, value, loc);
  }, *this, value, loc);
}
void Val::divAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Div, value, loc);
  }, *this, value, loc);
}
void Val::remAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Rem, value, loc);
  }, *this, value, loc);
}
void Val::bitAndAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::BitAnd, value, loc);
  }, *this, value, loc);
}
void Val::bitOrAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::BitOr, value, loc);
  }, *this, value, loc);
}
void Val::bitXorAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::BitXor, value, loc);
  }, *this, value, loc);
}
void Val::shlAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Shl, value, loc);
  }, *this, value, loc);
}
void Val::shrAssign(const ValueLike& value, Loc loc) const {
  return detail::guard([&] {
    compound(BinaryOp::Shr, value, loc);
  }, *this, value, loc);
}
void Val::increment(Loc loc) const {
  return detail::guard([&] { compound(BinaryOp::Add, 1, loc); }, *this, loc);
}
void Val::decrement(Loc loc) const {
  return detail::guard([&] { compound(BinaryOp::Sub, 1, loc); }, *this, loc);
}

void Val::compound(BinaryOp op, const ValueLike& value, const Loc& loc) const {
  return detail::guard([&] {
    assertAssignable(*node_);
    // binary() checks and widens the operands; the result must keep the target type
    const Val result = binary(op, *this, value);
    // a placeholder has no operation to take the operand from
    if (detail::poisoned(result)) return;
    if (!typesEqual(result.type(), type()))
      throw ShaderError(ErrorCode::TypeMismatch, std::format("\"{}=\" would change {} into {}", opText(op), typeName(type()),
                                    typeName(result.type())));
    const auto& b = std::get<ex::Binary>(result.node()->node);
    pushStatement(st::Assign{node_, b.b, op}, loc);
  }, *this, op, value, loc);
}

} // namespace moga
