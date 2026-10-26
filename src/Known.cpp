#include "Known.hpp"

#include "BuiltinTable.hpp"
#include "Half.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <format>
#include <optional>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <string_view>

#if defined(_MSC_VER)
#pragma fenv_access(on)
#elif defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif

namespace moga::detail {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

using Ranges = std::vector<KnownRange>;

KnownRange point(double d) { return {d, d}; }

bool isPoint(const KnownRange& r) { return r.low == r.high; }

KnownRange hull(const KnownRange& a, const KnownRange& b) {
  if (std::isnan(a.low) || std::isnan(b.low)) return point(kNan);
  return {std::min(a.low, b.low), std::max(a.high, b.high)};
}

std::optional<Ranges> known(const Expr& e);
std::optional<Ranges> operand(const Expr& e);

/** The f32 neighbor of v below it (up = false) or above it, found the same way as halfToward(). */
double f32Toward(double v, bool up) {
  using L = std::numeric_limits<float>;
  if (!std::isfinite(v) || v == 0.0) return v;
  // L::digits counts the hidden bit; L::min_exponent - 1 is the exponent of the smallest normal
  const int e = std::max(std::ilogb(std::abs(v)), L::min_exponent - 1);
  const double quantum = std::ldexp(1.0, e - (L::digits - 1));
  const double r = (up ? std::ceil(v / quantum) : std::floor(v / quantum)) * quantum;
  const auto max = static_cast<double>(L::max());
  if (r > max) return up ? kInf : max;
  if (r < -max) return up ? -max : -kInf;
  return r;
}

/** The smallest normal value of a float type. */
double minNormal(ScalarKind k) {
  // binary16 exponents start at -14
  return k == ScalarKind::F16 ? std::ldexp(1.0, -14) : static_cast<double>(std::numeric_limits<float>::min());
}

/**
 * The values a float type may store for an exact result in r. WGSL lets an
 * inexact result be either neighbor of the exact value, and lets a subnormal
 * be flushed to zero.
 */
KnownRange storeFloat(const KnownRange& r, ScalarKind k, bool mayFlush = true) {
  if (std::isnan(r.low) || std::isnan(r.high)) return point(kNan);
  const auto toward = [&](double v, bool up) { return k == ScalarKind::F16 ? halfToward(v, up) : f32Toward(v, up); };
  KnownRange out{toward(r.low, false), toward(r.high, true)};
  // an overflow is reported, never narrowed back to the largest value
  if (std::isinf(r.low)) out.low = r.low;
  if (std::isinf(r.high)) out.high = r.high;
  const double m = minNormal(k);
  const bool subnormal = out.low < m && out.high > -m && !(out.low == 0 && out.high == 0);
  if (subnormal && mayFlush) out = {std::min(out.low, 0.0), std::max(out.high, 0.0)};
  return out;
}

/**
 * The exact result of a + b, a - b, a * b or a / b lies in the returned
 * range: the double result, widened by one double step toward the rounding
 * error when the error free check finds one.
 */
KnownRange exactOp(BinaryOp op, double a, double b) {
  double r = 0;
  double error = 0;
  switch (op) {
  case BinaryOp::Add:
  case BinaryOp::Sub: {
    const double c = op == BinaryOp::Sub ? -b : b;
    r = a + c;
    // TwoSum: the exact error of the rounded sum
    const double cc = r - a;
    error = (a - (r - cc)) + (c - cc);
    break;
  }
  case BinaryOp::Mul:
    r = a * b;
    error = std::fma(a, b, -r);
    break;
  default:
    r = a / b;
    // the remainder a - r * b is exact; with b > 0 the quotient is above r when it is positive
    error = std::isfinite(r) && b != 0 ? -std::fma(r, b, -a) * (b < 0 ? -1.0 : 1.0) : 0.0;
    break;
  }
  if (std::isnan(r)) return point(kNan);
  if (!std::isfinite(r) || !std::isfinite(error) || error == 0) return point(r);
  return error > 0 ? KnownRange{r, std::nextafter(r, kInf)} : KnownRange{std::nextafter(r, -kInf), r};
}

/** The next value of the type above (up) or below v. */
double stepFloat(double v, bool up, ScalarKind k) {
  if (!std::isfinite(v)) return v;
  const double beyond = std::nextafter(v, up ? kInf : -kInf);
  return k == ScalarKind::F16 ? halfToward(beyond, up) : f32Toward(beyond, up);
}

/**
 * WGSL allows a float division 2.5 ULP from the exact quotient: two more
 * steps of the type on each side of the stored neighbors.
 */
KnownRange widenDivision(KnownRange r, ScalarKind k) {
  if (isPoint(r) && std::isfinite(r.low) && r.low == 0) return r;
  constexpr int kExtraSteps = 2;
  for (int i = 0; i < kExtraSteps; ++i) r = {stepFloat(r.low, false, k), stepFloat(r.high, true, k)};
  return r;
}

/** The range of a float operation over every pair of endpoints. */
KnownRange floatOp(BinaryOp op, const KnownRange& x, const KnownRange& y) {
  // a divisor that can be zero gives an infinity or NaN, which the non finite check reports
  if (op == BinaryOp::Div && y.low <= 0 && y.high >= 0) {
    if (isPoint(x) && isPoint(y)) return point(x.low / y.low);
    return {-kInf, kInf};
  }
  std::optional<KnownRange> out;
  for (const double a : {x.low, x.high})
    for (const double b : {y.low, y.high}) {
      const KnownRange r = exactOp(op, a, b);
      out = out ? hull(*out, r) : r;
    }
  return *out;
}

/** << and >> of 32 bit integers; binary() rejects amounts of 32 or more and lost bits. */
std::optional<double> shiftOp(BinaryOp op, std::int64_t a, std::int64_t b, bool isUnsigned) {
  if (b < 0 || b >= 32) return std::nullopt;
  const auto bits = static_cast<std::uint32_t>(a);
  if (op == BinaryOp::Shl) return static_cast<double>(static_cast<std::uint32_t>(bits << b));
  // >> is logical for u32 and arithmetic for i32
  if (isUnsigned) return static_cast<double>(bits >> b);
  return static_cast<double>(static_cast<std::int32_t>(bits) >> b);
}

/**
 * i32 and u32 arithmetic in 64 bit integers. The operands are already in 32
 * bit range; the product wraps modulo 2^64, which keeps its low 32 bits exact.
 * Bit operations work on the two's complement bits, which evaluate() then
 * reads in the result type.
 */
std::optional<double> integerOp(BinaryOp op, double l, double r, bool isUnsigned) {
  const auto a = static_cast<std::int64_t>(l);
  const auto b = static_cast<std::int64_t>(r);
  switch (op) {
  case BinaryOp::Add:
    return static_cast<double>(a + b);
  case BinaryOp::Sub:
    return static_cast<double>(a - b);
  case BinaryOp::Mul:
    return static_cast<double>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b)));
  case BinaryOp::Div:
    // truncates toward zero; binary() rejects a known zero divisor
    if (b == 0) return std::nullopt;
    return static_cast<double>(a / b);
  case BinaryOp::Rem:
    // the sign of the dividend, as WGSL
    if (b == 0) return std::nullopt;
    return static_cast<double>(a % b);
  case BinaryOp::BitAnd:
    return static_cast<double>(a & b);
  case BinaryOp::BitOr:
    return static_cast<double>(a | b);
  case BinaryOp::BitXor:
    return static_cast<double>(a ^ b);
  case BinaryOp::Shl:
  case BinaryOp::Shr:
    return shiftOp(op, a, b, isUnsigned);
  default:
    return std::nullopt;
  }
}

/**
 * Integer operands that are ranges come from float conversions. + and - are
 * exact on the endpoints; other operations need single values. evaluate()
 * then wraps the result into the type, or gives up when the range wraps.
 */
std::optional<KnownRange> integerRangeOp(BinaryOp op, const KnownRange& x, const KnownRange& y, bool isUnsigned) {
  if (isPoint(x) && isPoint(y)) {
    const auto v = integerOp(op, x.low, y.low, isUnsigned);
    return v ? std::optional(point(*v)) : std::nullopt;
  }
  if (op == BinaryOp::Add) return KnownRange{x.low + y.low, x.high + y.high};
  if (op == BinaryOp::Sub) return KnownRange{x.low - y.high, x.high - y.low};
  return std::nullopt;
}

/** Whether l < r for every pair of values (1), for none (0), or for some only. */
KnownRange less(const KnownRange& l, const KnownRange& r) {
  if (l.high < r.low) return point(1);
  if (l.low >= r.high) return point(0);
  return {0, 1};
}

KnownRange negate(const KnownRange& b) { return {1.0 - b.high, 1.0 - b.low}; }

/** Comparisons and logic give a bool, 1 for true and 0 for false, or both when the operands are ranges. */
std::optional<KnownRange> boolOp(BinaryOp op, const KnownRange& l, const KnownRange& r) {
  switch (op) {
  case BinaryOp::Lt:
    return less(l, r);
  case BinaryOp::Gt:
    return less(r, l);
  case BinaryOp::Ge:
    return negate(less(l, r));
  case BinaryOp::Le:
    return negate(less(r, l));
  case BinaryOp::Eq:
  case BinaryOp::Ne: {
    const bool disjoint = l.high < r.low || r.high < l.low;
    const bool same = isPoint(l) && isPoint(r) && l.low == r.low;
    KnownRange eq{0, 1};
    if (disjoint) eq = point(0);
    if (same) eq = point(1);
    return op == BinaryOp::Eq ? eq : negate(eq);
  }
  case BinaryOp::And:
    return KnownRange{std::min(l.low, r.low), std::min(l.high, r.high)};
  case BinaryOp::Or:
    return KnownRange{std::max(l.low, r.low), std::max(l.high, r.high)};
  default:
    return std::nullopt;
  }
}

bool isFloatArithmetic(BinaryOp op) {
  return op == BinaryOp::Add || op == BinaryOp::Sub || op == BinaryOp::Mul || op == BinaryOp::Div;
}

/** Arithmetic, comparisons and integer bit operations, component by component; a scalar applies to each component. */
std::optional<Ranges> knownArithmetic(const ex::Binary& b) {
  if (b.a->type->kind == TypeKind::Matrix || b.b->type->kind == TypeKind::Matrix) return std::nullopt;
  const auto x = operand(*b.a);
  const auto y = operand(*b.b);
  if (!x || !y) return std::nullopt;
  const std::size_t n = std::max(x->size(), y->size());
  Ranges out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const KnownRange& l = (*x)[x->size() == 1 ? 0 : i];
    const KnownRange& r = (*y)[y->size() == 1 ? 0 : i];
    std::optional<KnownRange> v = boolOp(b.op, l, r);
    if (!v && isInteger(b.a->type)) v = integerRangeOp(b.op, l, r, b.a->type->scalar == ScalarKind::U32);
    else if (!v && isFloatArithmetic(b.op)) v = floatOp(b.op, l, r);
    if (!v) return std::nullopt;
    out[i] = *v;
  }
  return out;
}

/** The largest f32 that is not above the maximum of T: float(max) rounds up to a power of two. */
template <class T> double largestF32Below() {
  const auto max = static_cast<float>(std::numeric_limits<T>::max());
  return static_cast<double>(std::nextafter(max, 0.0f));
}

/** bool(x): true when no value in the range is zero, false when it is exactly zero. */
KnownRange toBool(const KnownRange& r) {
  if (r.low == 0 && r.high == 0) return point(0);
  if (r.low > 0 || r.high < 0) return point(1);
  return {0, 1};
}

/**
 * A scalar conversion, for example i32(f). A float becomes an integer by
 * clamping and truncation, as WGSL defines it, which is the same for every
 * value of a range that truncates to one integer. evaluate() stores a float
 * result in its type.
 */
std::optional<Ranges> knownConversion(const Expr& from, ScalarKind to) {
  auto parts = known(from);
  if (!parts) return parts;
  if (to == ScalarKind::Bool) {
    for (KnownRange& r : *parts) r = toBool(r);
    return parts;
  }
  // u32(i32) and i32(u32) keep the bits, which evaluate() wraps
  if (isFloatKind(to) || isInteger(from.type)) return parts;
  // the limits are the f32 values closest to the range ends that are still inside it
  const double low = to == ScalarKind::U32 ? 0.0 : static_cast<double>(std::numeric_limits<std::int32_t>::min());
  const double high = to == ScalarKind::U32 ? largestF32Below<std::uint32_t>() : largestF32Below<std::int32_t>();
  for (KnownRange& r : *parts) {
    if (std::isnan(r.low)) return std::nullopt;
    r = {std::trunc(std::clamp(r.low, low, high)), std::trunc(std::clamp(r.high, low, high))};
  }
  return parts;
}

/** -x, !x and ~x: ~x is -x - 1, which evaluate() wraps for u32. */
std::optional<Ranges> knownUnary(const ex::Unary& u, const TypePtr& type) {
  if (u.op == UnaryOp::BitNot && !isInteger(type)) return std::nullopt;
  auto parts = operand(*u.a);
  if (!parts) return parts;
  for (KnownRange& r : *parts) {
    if (u.op == UnaryOp::Not) r = negate(r);
    else if (u.op == UnaryOp::Neg) r = {-r.high, -r.low};
    else r = {-r.high - 1.0, -r.low - 1.0};
  }
  return parts;
}

/** select(f, t, cond): t where cond holds, per component for a bool vector condition; both when cond can be either. */
std::optional<Ranges> knownSelect(const ex::Select& s) {
  const auto cond = known(*s.cond);
  const auto t = cond ? known(*s.t) : std::nullopt;
  const auto f = t ? known(*s.f) : std::nullopt;
  if (!f) return std::nullopt;
  Ranges out(t->size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const KnownRange& c = (*cond)[cond->size() == 1 ? 0 : i];
    if (isPoint(c)) out[i] = c.low != 0 ? (*t)[i] : (*f)[i];
    else out[i] = hull((*t)[i], (*f)[i]);
  }
  return out;
}

/** An index known as one integer. */
std::optional<double> knownIndexValue(const Expr& index) {
  const auto at = known(index);
  if (!at || !isPoint(at->front())) return std::nullopt;
  return at->front().low;
}

/**
 * The expression a value is made from: the value of a named constant, the
 * argument of a struct or array constructor that a field or a known index picks.
 */
const Expr& source(const Expr& e) {
  if (const auto* c = std::get_if<ex::Constant>(&e.node)) return source(*c->c->value);
  if (const auto* m = std::get_if<ex::Member>(&e.node)) {
    const Expr& base = source(*m->base);
    const auto* k = std::get_if<ex::Construct>(&base.node);
    if (k == nullptr || base.type->kind != TypeKind::Struct) return e;
    for (std::size_t i = 0; i < base.type->fields.size(); ++i)
      if (base.type->fields[i].name == m->field) return source(*k->args[i]);
    return e;
  }
  if (const auto* x = std::get_if<ex::Index>(&e.node)) {
    const Expr& base = source(*x->base);
    const auto* k = std::get_if<ex::Construct>(&base.node);
    const auto i = k != nullptr && base.type->kind == TypeKind::Array ? knownIndexValue(*x->index) : std::nullopt;
    if (!i || *i < 0 || *i >= static_cast<double>(k->args.size())) return e;
    return source(*k->args[static_cast<std::size_t>(*i)]);
  }
  return e;
}

/**
 * Column i of a matrix built from known values: from one scalar (the
 * diagonal), from columns, from every component in column order, or from
 * another matrix (a conversion, or mat3 from mat4).
 */
std::optional<Ranges> knownColumn(const Expr& m, double i) {
  const auto* k = std::get_if<ex::Construct>(&m.node);
  if (k == nullptr || i < 0 || i >= m.type->cols) return std::nullopt;
  const auto col = static_cast<std::size_t>(i);
  const auto rows = static_cast<std::size_t>(m.type->rows);
  if (k->args.size() == 1 && k->args[0]->type->kind == TypeKind::Scalar) {
    const auto d = known(*k->args[0]);
    if (!d) return std::nullopt;
    Ranges out(rows, point(0));
    if (col < rows) out[col] = d->front();
    return out;
  }
  if (k->args.size() == 1 && k->args[0]->type->kind == TypeKind::Matrix) {
    auto column = knownColumn(source(*k->args[0]), i);
    if (!column) return std::nullopt;
    // the top left part, each component stored in this matrix's type before anything reads it
    column->resize(rows);
    for (KnownRange& r : *column) r = storeFloat(r, m.type->scalar, false);
    return column;
  }
  if (k->args.size() == static_cast<std::size_t>(m.type->cols)) return known(*k->args[col]);
  if (k->args.size() != rows * static_cast<std::size_t>(m.type->cols)) return std::nullopt;
  Ranges out;
  for (std::size_t r = 0; r < rows; ++r) {
    const auto d = known(*k->args[col * rows + r]);
    if (!d) return std::nullopt;
    out.push_back(d->front());
  }
  return out;
}

/** Every component of a matrix built from known values, column by column. */
std::optional<Ranges> knownMatrix(const Expr& m) {
  Ranges out;
  for (int c = 0; c < m.type->cols; ++c) {
    const auto column = knownColumn(m, c);
    if (!column) return std::nullopt;
    out.insert(out.end(), column->begin(), column->end());
  }
  return out;
}

/** A field of a struct built from known values. */
std::optional<Ranges> knownMember(const ex::Member& m) {
  const Expr& base = source(*m.base);
  const auto* k = std::get_if<ex::Construct>(&base.node);
  if (k == nullptr || base.type->kind != TypeKind::Struct) return std::nullopt;
  for (std::size_t i = 0; i < base.type->fields.size(); ++i)
    if (base.type->fields[i].name == m.field) return known(*k->args[i]);
  return std::nullopt;
}

/** A component of a known vector or matrix, or an element of an array built from known values. */
std::optional<Ranges> knownIndex(const ex::Index& x) {
  const auto at = knownIndexValue(*x.index);
  if (!at) return std::nullopt;
  const Expr& base = source(*x.base);
  const double i = *at;
  if (base.type->kind == TypeKind::Vector) {
    const auto parts = known(base);
    if (!parts || i < 0 || i >= static_cast<double>(parts->size())) return std::nullopt;
    return Ranges{(*parts)[static_cast<std::size_t>(i)]};
  }
  if (base.type->kind == TypeKind::Matrix) return knownColumn(base, i);
  const auto* k = std::get_if<ex::Construct>(&base.node);
  if (k == nullptr || base.type->kind != TypeKind::Array || i < 0 || i >= static_cast<double>(k->args.size()))
    return std::nullopt;
  return known(*k->args[static_cast<std::size_t>(i)]);
}

/** Picks components of a known value; the DSL stores swizzles in xyzw letters. */
std::optional<Ranges> knownSwizzle(const ex::Swizzle& w) {
  const auto base = known(*w.base);
  if (!base) return std::nullopt;
  Ranges parts;
  for (const char c : w.comps) {
    const auto index = std::string_view("xyzw").find(c);
    if (index >= base->size()) return std::nullopt;
    parts.push_back((*base)[index]);
  }
  return parts;
}

/**
 * Applies a function that never decreases when an argument grows to each
 * component: the lows of the arguments give the low, the highs the high.
 * Scalar arguments apply to every component.
 */
template <class F> std::optional<Ranges> monotone(const std::vector<ExprPtr>& args, F f) {
  std::vector<Ranges> values;
  std::size_t n = 1;
  for (const ExprPtr& a : args) {
    auto v = operand(*a);
    if (!v) return std::nullopt;
    n = std::max(n, v->size());
    values.push_back(std::move(*v));
  }
  Ranges out(n);
  std::vector<double> lows(values.size());
  std::vector<double> highs(values.size());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < values.size(); ++k) {
      const KnownRange& r = values[k][values[k].size() == 1 ? 0 : i];
      lows[k] = r.low;
      highs[k] = r.high;
    }
    out[i] = {f(lows), f(highs)};
  }
  return out;
}

/** abs(): the range folds at zero. */
KnownRange absRange(const KnownRange& r) {
  if (r.low >= 0) return r;
  if (r.high <= 0) return {-r.high, -r.low};
  return {0, std::max(-r.low, r.high)};
}

/** fract(x) = x - floor(x), stored in its type by evaluate(); a range across an integer spans [0, 1]. */
KnownRange fractRange(const KnownRange& r) {
  const double below = std::floor(r.low);
  if (std::floor(r.high) != below) return {0, 1};
  return hull(exactOp(BinaryOp::Sub, r.low, below), exactOp(BinaryOp::Sub, r.high, below));
}

/** dot() of float vectors: every product and partial sum may round, in any order of the sum. */
KnownRange floatDot(const Ranges& a, const Ranges& b, ScalarKind k) {
  Ranges products;
  for (std::size_t i = 0; i < a.size(); ++i) products.push_back(storeFloat(floatOp(BinaryOp::Mul, a[i], b[i]), k));
  std::vector<std::size_t> order(products.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::optional<KnownRange> out;
  do {
    KnownRange sum = point(0);
    for (const std::size_t i : order) sum = storeFloat(floatOp(BinaryOp::Add, sum, products[i]), k);
    out = out ? hull(*out, sum) : sum;
  } while (std::ranges::next_permutation(order).found);
  return *out;
}

/** dot(): integer products and sum wrap modulo 2^32, as WGSL computes them. */
std::optional<Ranges> knownDot(const ex::Call& c) {
  const auto a = operand(*c.args[0]);
  const auto b = a ? operand(*c.args[1]) : std::nullopt;
  if (!b) return std::nullopt;
  if (!isInteger(c.args[0]->type)) return Ranges{floatDot(*a, *b, c.args[0]->type->scalar)};
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < a->size(); ++i) {
    if (!isPoint((*a)[i]) || !isPoint((*b)[i])) return std::nullopt;
    sum += static_cast<std::uint32_t>(static_cast<std::int64_t>((*a)[i].low)) *
           static_cast<std::uint32_t>(static_cast<std::int64_t>((*b)[i].low));
  }
  // evaluate() wraps the result into the i32 or u32 range
  return Ranges{point(static_cast<double>(sum))};
}

/** any() (every = false) or all() (every = true) of a bool vector or scalar. */
std::optional<Ranges> knownReduce(const ex::Call& c, bool every) {
  const auto v = known(*c.args[0]);
  if (!v) return std::nullopt;
  KnownRange out = point(every ? 1 : 0);
  for (const KnownRange& r : *v)
    out = every ? KnownRange{std::min(out.low, r.low), std::min(out.high, r.high)}
                : KnownRange{std::max(out.low, r.low), std::max(out.high, r.high)};
  return Ranges{out};
}

/** Applies f to each component range of the single argument. */
template <class F> std::optional<Ranges> each(const ExprPtr& arg, F f) {
  auto v = operand(*arg);
  if (v)
    for (KnownRange& r : *v) r = f(r);
  return v;
}

/**
 * The constant evaluable builtins whose possible results the evaluator can
 * bound. Others (sin, pow, ...) stay unknown.
 */
std::optional<Ranges> knownCall(const ex::Call& c) {
  using A = const std::vector<double>&;
  switch (c.fn) {
  case BuiltinFunction::Min:
    return monotone(c.args, [](A a) { return std::min(a[0], a[1]); });
  case BuiltinFunction::Max:
    return monotone(c.args, [](A a) { return std::max(a[0], a[1]); });
  case BuiltinFunction::Clamp:
    return monotone(c.args, [](A a) { return std::min(std::max(a[0], a[1]), a[2]); });
  case BuiltinFunction::Abs:
    return each(c.args[0], absRange);
  case BuiltinFunction::Sign:
    return monotone(c.args, [](A a) { return static_cast<double>((a[0] > 0) - (a[0] < 0)); });
  case BuiltinFunction::Floor:
    return monotone(c.args, [](A a) { return std::floor(a[0]); });
  case BuiltinFunction::Ceil:
    return monotone(c.args, [](A a) { return std::ceil(a[0]); });
  case BuiltinFunction::Trunc:
    return monotone(c.args, [](A a) { return std::trunc(a[0]); });
  case BuiltinFunction::Round:
    return monotone(c.args, [](A a) { return roundHalfToEven(a[0]); });
  case BuiltinFunction::Fract:
    return each(c.args[0], fractRange);
  case BuiltinFunction::Dot:
    return knownDot(c);
  case BuiltinFunction::Any:
    return knownReduce(c, false);
  case BuiltinFunction::All:
    return knownReduce(c, true);
  default:
    return std::nullopt;
  }
}

std::optional<Ranges> knownConstruct(const Expr& e, const ex::Construct& c);

/** The value of the node before it is stored in its own type. */
std::optional<Ranges> unrounded(const Expr& e) {
  if (const auto* l = std::get_if<ex::Lit>(&e.node)) {
    if (const double* d = std::get_if<double>(&l->value)) return Ranges{point(*d)};
    return Ranges{point(std::get<bool>(l->value) ? 1.0 : 0.0)};
  }
  if (const auto* s = std::get_if<ex::Select>(&e.node)) return knownSelect(*s);
  if (const auto* c = std::get_if<ex::Constant>(&e.node)) return known(*c->c->value);
  if (const auto* u = std::get_if<ex::Unary>(&e.node)) return knownUnary(*u, e.type);
  if (const auto* b = std::get_if<ex::Binary>(&e.node)) return knownArithmetic(*b);
  if (const auto* w = std::get_if<ex::Swizzle>(&e.node)) return knownSwizzle(*w);
  if (const auto* m = std::get_if<ex::Member>(&e.node)) return knownMember(*m);
  if (const auto* x = std::get_if<ex::Index>(&e.node)) return knownIndex(*x);
  if (const auto* c = std::get_if<ex::Call>(&e.node)) return knownCall(*c);
  if (const auto* k = std::get_if<ex::Construct>(&e.node)) return knownConstruct(e, *k);
  return std::nullopt;
}

/** A scalar conversion, a vector from components, or a matrix. */
std::optional<Ranges> knownConstruct(const Expr& e, const ex::Construct& c) {
  const ex::Construct* k = &c;
  if (e.type->kind == TypeKind::Matrix) return knownMatrix(e);
  if (e.type->kind != TypeKind::Vector && e.type->kind != TypeKind::Scalar) return std::nullopt;
  if (e.type->kind == TypeKind::Scalar) return knownConversion(*k->args.front(), e.type->scalar);
  Ranges parts;
  for (const ExprPtr& arg : k->args) {
    // vec2i(vec2f(7.9, 0.0)) converts each component: (7, 0)
    const auto inner = arg->type->scalar == e.type->scalar ? known(*arg)
                                                           : knownConversion(*arg, e.type->scalar);
    if (!inner) return std::nullopt;
    parts.insert(parts.end(), inner->begin(), inner->end());
  }
  // one scalar fills every component
  if (parts.size() == 1) parts.assign(static_cast<std::size_t>(e.type->size), parts.front());
  return parts;
}

/**
 * The results of one knownRanges() call. A reused value is a graph of
 * shared nodes, which a plain recursion would expand into a tree.
 */
thread_local std::unordered_map<const Expr*, std::optional<Ranges>>* memo = nullptr;

std::optional<Ranges> evaluate(const Expr& e);

std::optional<Ranges> known(const Expr& e) {
  if (const auto it = memo->find(&e); it != memo->end()) return it->second;
  auto parts = evaluate(e);
  memo->emplace(&e, parts);
  return parts;
}

/**
 * The value of an operand of an operation or a builtin: WGSL may read a
 * subnormal input as zero.
 */
std::optional<Ranges> operand(const Expr& e) {
  auto parts = known(e);
  if (!parts || !isFloatKind(e.type->scalar)) return parts;
  const double m = minNormal(e.type->scalar);
  for (KnownRange& r : *parts)
    if (r.low < m && r.high > -m && !(r.low == 0 && r.high == 0)) r = {std::min(r.low, 0.0), std::max(r.high, 0.0)};
  return parts;
}

/**
 * i32 or u32 wrapping of a range. When its ends land in different turns of
 * 2^32 the wrapped values are two pieces at both ends of the type; the range
 * covers the whole type, so a check sees both.
 */
std::optional<KnownRange> wrapInteger(const KnownRange& r, ScalarKind k) {
  const double turn = std::ldexp(1.0, std::numeric_limits<std::uint32_t>::digits);
  const double min = k == ScalarKind::U32 ? 0.0 : static_cast<double>(std::numeric_limits<std::int32_t>::min());
  const double turnLow = std::floor((r.low - min) / turn);
  if (turnLow != std::floor((r.high - min) / turn)) return KnownRange{min, min + turn - 1};
  return KnownRange{r.low - turnLow * turn, r.high - turnLow * turn};
}

std::optional<Ranges> evaluate(const Expr& e) {
  auto parts = unrounded(e);
  if (!parts) return parts;
  // WGSL evaluates constant expressions in the concrete type: a float may store either neighbor
  // of an inexact result (storeFloat), i32 and u32 wrap modulo 2^32 (two's complement for i32)
  // a literal or a conversion is never flushed; an operation or a builtin result may be
  const bool computed = std::holds_alternative<ex::Binary>(e.node) || std::holds_alternative<ex::Unary>(e.node) ||
                        std::holds_alternative<ex::Call>(e.node);
  const auto* binary = std::get_if<ex::Binary>(&e.node);
  const bool division = binary != nullptr && binary->op == BinaryOp::Div && isFloatKind(e.type->scalar);
  for (KnownRange& r : *parts) {
    if (isFloatKind(e.type->scalar)) {
      r = storeFloat(r, e.type->scalar, computed);
      if (division) r = widenDivision(r, e.type->scalar);
      continue;
    }
    if (e.type->scalar == ScalarKind::Bool) continue;
    const auto wrapped = wrapInteger(r, e.type->scalar);
    if (!wrapped) return std::nullopt;
    r = *wrapped;
  }
  return parts;
}

} // namespace

/**
 * The evaluator relies on round to nearest for its error free checks, so
 * the host rounding mode is set for the evaluation and restored afterwards.
 */
class NearestRounding {
public:
  NearestRounding() : previous_(std::fegetround()) { std::fesetround(FE_TONEAREST); }
  ~NearestRounding() { std::fesetround(previous_); }
  NearestRounding(const NearestRounding&) = delete;
  NearestRounding& operator=(const NearestRounding&) = delete;

private:
  int previous_;
};

std::optional<std::vector<KnownRange>> knownRanges(const Expr& e) {
  // a local, an input or a memory read below makes the value unknown; most run time values stop here
  if (!e.constantLeaves) return std::nullopt;
  const NearestRounding nearest;
  std::unordered_map<const Expr*, std::optional<Ranges>> results;
  memo = &results;
  struct Reset {
    ~Reset() { memo = nullptr; }
  } reset;
  return known(e);
}

bool sameValue(const Expr& a, const Expr& b) {
  // a node used twice is printed twice, and each printed copy may round its own way
  const auto* ca = std::get_if<ex::Constant>(&a.node);
  const auto* cb = std::get_if<ex::Constant>(&b.node);
  if (ca != nullptr && cb != nullptr) return ca->c == cb->c;
  // a scalar spread to a vector builds a new constructor at each use
  const auto* ka = std::get_if<ex::Construct>(&a.node);
  const auto* kb = std::get_if<ex::Construct>(&b.node);
  if (ka == nullptr || kb == nullptr || a.type != b.type || ka->args.size() != kb->args.size()) return false;
  for (std::size_t i = 0; i < ka->args.size(); ++i)
    if (!sameValue(*ka->args[i], *kb->args[i])) return false;
  return true;
}

std::string describe(const KnownRange& r) {
  if (isPoint(r)) return std::format("{}", r.low);
  return std::format("{} to {}", r.low, r.high);
}

std::optional<std::vector<double>> knownComponents(const Expr& e) {
  const auto ranges = knownRanges(e);
  if (!ranges) return std::nullopt;
  std::vector<double> values;
  for (const KnownRange& r : *ranges) {
    // NaN is one value, though NaN != NaN
    if (!isPoint(r) && !std::isnan(r.low)) return std::nullopt;
    values.push_back(r.low);
  }
  return values;
}

namespace {

bool constForm(const Expr& e, bool allowRuntimeConstants, std::unordered_map<const Expr*, bool>& seen);

/** Each shared node is checked once. */
bool constCached(const Expr& e, bool allowRuntimeConstants, std::unordered_map<const Expr*, bool>& seen) {
  if (const auto it = seen.find(&e); it != seen.end()) return it->second;
  const bool result = constForm(e, allowRuntimeConstants, seen);
  seen.emplace(&e, result);
  return result;
}

bool constForm(const Expr& e, bool allowRuntimeConstants, std::unordered_map<const Expr*, bool>& seen) {
  const auto all = [&](const std::vector<ExprPtr>& args) {
    return std::ranges::all_of(args, [&](const ExprPtr& a) { return constCached(*a, allowRuntimeConstants, seen); });
  };
  return std::visit(
      Overloaded{
          [](const ex::Lit&) { return true; },
          [&](const ex::Constant& c) { return allowRuntimeConstants || isConstExpression(*c.c->value); },
          [&](const ex::Construct& c) { return all(c.args); },
          [&](const ex::Call& c) {
            const BuiltinSpec& spec = builtinSpec(c.fn);
            return spec.constEval && (allowRuntimeConstants || !spec.bitcast) && all(c.args);
          },
          [&](const ex::Unary& u) { return all({u.a}); },
          [&](const ex::Binary& b) { return all({b.a, b.b}); },
          [&](const ex::Swizzle& s) { return all({s.base}); },
          [&](const ex::Select& s) { return all({s.cond, s.t, s.f}); },
          [&](const ex::Member& m) { return all({m.base}); },
          [&](const ex::Index& x) { return all({x.base, x.index}); },
          [](const auto&) { return false; },
      },
      e.node);
}

} // namespace

bool isConstExpression(const Expr& e, bool allowRuntimeConstants) {
  if (!e.constantLeaves) return false;
  std::unordered_map<const Expr*, bool> seen;
  return constCached(e, allowRuntimeConstants, seen);
}

std::optional<std::string> nonFiniteConstant(const Expr& e) {
  if (!e.constantLeaves || !isFloat(e.type)) return std::nullopt;
  const auto parts = knownRanges(e);
  if (!parts) return std::nullopt;
  // one possible result that is not finite is enough for WGSL to reject the shader somewhere
  for (const KnownRange& r : *parts) {
    if (std::isfinite(r.low) && std::isfinite(r.high)) continue;
    std::string_view what = "is infinite";
    if (std::isnan(r.low)) what = "is NaN";
    else if (r.low != r.high) what = "can be infinite";
    return std::format("the constant value {} in {}, which WGSL rejects", what, typeName(scalarType(e.type->scalar)));
  }
  return std::nullopt;
}

} // namespace moga::detail
