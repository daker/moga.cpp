#include "BuiltinTable.hpp"
#include "Context.hpp"
#include "Known.hpp"
#include "moga/dsl/Math.hpp"
#include "moga/dsl/Constructors.hpp"
#include "moga/dsl/Statements.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <ranges>
#include <vector>

namespace moga {

using namespace detail;

namespace {

using F = BuiltinFunction;

/** Constant arguments outside the domain of a builtin; WGSL rejects the call (Tint, Chrome 155). */
struct DomainRule {
  F fn;
  bool (*outside)(double x, double y);
};

constexpr std::array kDomainRules = {
    DomainRule{F::Sqrt, [](double x, double) { return x < 0; }},
    DomainRule{F::InverseSqrt, [](double x, double) { return x <= 0; }},
    DomainRule{F::Log, [](double x, double) { return x <= 0; }},
    DomainRule{F::Log2, [](double x, double) { return x <= 0; }},
    DomainRule{F::Asin, [](double x, double) { return std::abs(x) > 1; }},
    DomainRule{F::Acos, [](double x, double) { return std::abs(x) > 1; }},
    DomainRule{F::Acosh, [](double x, double) { return x < 1; }},
    DomainRule{F::Atanh, [](double x, double) { return std::abs(x) >= 1; }},
    DomainRule{F::Pow, [](double x, double y) { return x < 0 || (x == 0 && y <= 0); }},
};

/**
 * Builtins whose constant result can overflow. The host math only decides
 * whether the result is finite in the type; it never becomes a known value.
 */
struct OverflowRule {
  F fn;
  double (*value)(double x);
};

constexpr std::array kOverflowRules = {
    OverflowRule{F::Exp, [](double x) { return std::exp(x); }},
    OverflowRule{F::Exp2, [](double x) { return std::exp2(x); }},
    OverflowRule{F::Sinh, [](double x) { return std::sinh(x); }},
    OverflowRule{F::Cosh, [](double x) { return std::cosh(x); }},
    OverflowRule{F::Degrees, [](double x) { return x * (180.0 / std::numbers::pi); }},
};

bool overflows(double v, ScalarKind kind) {
  const double limit = kind == ScalarKind::F16 ? kF16Max : kF32Max;
  return !(std::abs(v) <= limit);
}

/** Known component i of argument k; a scalar applies to every component. */
double at(const std::vector<std::vector<double>>& known, std::size_t k, std::size_t i) {
  return known[k][known[k].size() == 1 ? 0 : i];
}

/**
 * The products, squares, sums and differences that a vector builtin computes
 * from known arguments; WGSL rejects a constant call when one of them
 * overflows (Tint, Chrome 155: length(vec2f(1e20f, 0.0f)), whose square
 * overflows, as well as pow(10.0f, 39.0f) and dot of 2e19 by itself).
 */
std::vector<double> vectorSteps(F fn, const std::vector<std::vector<double>>& known) {
  std::vector<double> steps;
  const std::size_t n = known[0].size();
  const auto arg = [&](std::size_t k, std::size_t i) { return at(known, k, i); };
  // WGSL may add the products in any order: the largest partial sums are all the positive
  // products together and all the negative ones together
  const auto dot = [&](std::size_t x, std::size_t y) {
    double sum = 0;
    double positive = 0;
    double negative = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double p = arg(x, i) * arg(y, i);
      steps.push_back(p);
      sum += p;
      (p > 0 ? positive : negative) += p;
    }
    steps.insert(steps.end(), {positive, negative, sum});
    return sum;
  };
  switch (fn) {
  case F::Pow:
    for (std::size_t i = 0; i < n; ++i) steps.push_back(std::pow(arg(0, i), arg(1, i)));
    break;
  case F::Dot:
    dot(0, 1);
    break;
  case F::Length:
  case F::Normalize:
    steps.push_back(std::sqrt(dot(0, 0)));
    break;
  case F::Distance: {
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = arg(0, i) - arg(1, i);
      steps.insert(steps.end(), {d, d * d, sum += d * d});
    }
    steps.push_back(std::sqrt(sum));
    break;
  }
  case F::Cross:
    for (std::size_t i = 0; i < 3; ++i) {
      const double p = arg(0, (i + 1) % 3) * arg(1, (i + 2) % 3);
      const double q = arg(0, (i + 2) % 3) * arg(1, (i + 1) % 3);
      steps.insert(steps.end(), {p, q, p - q});
    }
    break;
  case F::Mix:
    for (std::size_t i = 0; i < n; ++i) {
      const double x = arg(0, i) * (1.0 - arg(2, i));
      const double y = arg(1, i) * arg(2, i);
      steps.insert(steps.end(), {x, y, x + y});
    }
    break;
  case F::Reflect: {
    // e1 - 2 * dot(e2, e1) * e2
    const double d = dot(1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      const double t = 2.0 * d * arg(1, i);
      steps.insert(steps.end(), {t, arg(0, i) - t});
    }
    break;
  }
  default:
    break;
  }
  return steps;
}

void requireKnownDomain(F fn, const TypePtr& type, const std::vector<Val>& args,
                        const std::vector<std::vector<double>>& known);

/** The domain checks on the lows, the highs and the values of largest magnitude. */
void requireDomainAtPicks(F fn, const TypePtr& type, const std::vector<Val>& args,
                          const std::vector<std::vector<KnownRange>>& ranges) {
  using Pick = double (*)(const KnownRange&);
  const std::array<Pick, 3> picks{
      [](const KnownRange& r) { return r.low; }, [](const KnownRange& r) { return r.high; },
      [](const KnownRange& r) { return std::abs(r.low) > std::abs(r.high) ? r.low : r.high; }};
  for (const Pick pick : picks) {
    std::vector<std::vector<double>> known;
    for (const auto& arg : ranges) known.push_back(arg | std::views::transform(pick) | std::ranges::to<std::vector>());
    requireKnownDomain(fn, type, args, known);
  }
}

/**
 * Throws when every argument is known and the call is outside the domain or
 * overflows for some of the values WGSL may give the arguments. The checks
 * run on the lows, the highs and the values of largest magnitude.
 */
void requireConstantDomain(F fn, const TypePtr& type, const std::vector<Val>& args) {
  if (!isFloat(type) && fn != F::Length) return;
  std::vector<std::vector<KnownRange>> ranges;
  for (const Val& a : args) {
    auto v = knownRanges(*a.node());
    if (!v) return;
    ranges.push_back(std::move(*v));
  }
  if (ranges.empty()) return;
  // the zero vector is possible when every component can be zero
  if (fn == F::Normalize && std::ranges::all_of(ranges[0], [](const KnownRange& r) { return mayBe(r, 0); }))
    throw ShaderError(ErrorCode::InvalidConstant, "normalize: the constant vector can be zero");
  // the steps are products and sums of the arguments, largest at the ends of the ranges: every
  // combination of ends, each component chosen on its own as WGSL may
  std::vector<std::pair<std::size_t, std::size_t>> open;
  for (std::size_t a = 0; a < ranges.size(); ++a)
    for (std::size_t i = 0; i < ranges[a].size(); ++i)
      if (!isExact(ranges[a][i])) open.emplace_back(a, i);
  std::vector<std::vector<double>> known;
  for (const auto& arg : ranges)
    known.push_back(arg | std::views::transform(&KnownRange::low) | std::ranges::to<std::vector>());
  // past this many uncertain components, the lows, the highs and the largest magnitudes stand for all
  constexpr std::size_t kMaxOpen = 12;
  if (open.size() > kMaxOpen) {
    requireDomainAtPicks(fn, type, args, ranges);
    return;
  }
  for (std::uint32_t mask = 0; mask < (1u << open.size()); ++mask) {
    for (std::size_t b = 0; b < open.size(); ++b) {
      const auto [a, i] = open[b];
      known[a][i] = (mask >> b) & 1 ? ranges[a][i].high : ranges[a][i].low;
    }
    requireKnownDomain(fn, type, args, known);
  }
}

void requireKnownDomain(F fn, const TypePtr& type, const std::vector<Val>& args,
                        const std::vector<std::vector<double>>& known) {
  const std::string_view name = builtinName(fn);
  const std::size_t n = known[0].size();
  const auto domain = std::ranges::find(kDomainRules, fn, &DomainRule::fn);
  const auto overflow = std::ranges::find(kOverflowRules, fn, &OverflowRule::fn);
  for (std::size_t i = 0; i < n; ++i) {
    const double y = known.size() > 1 ? at(known, 1, i) : 0.0;
    if (domain != kDomainRules.end() && domain->outside(known[0][i], y))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("{}: the constant argument {} is outside its domain", name, known[0][i]));
    if (overflow != kOverflowRules.end() && overflows(overflow->value(known[0][i]), type->scalar))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("{}({}) overflows {}", name, known[0][i], typeName(scalarType(type->scalar))));
  }
  double squares = 0;
  for (const double c : known[0]) squares += c * c;
  if (fn == F::Normalize && squares == 0) throw ShaderError(ErrorCode::InvalidConstant, "normalize: the constant vector is zero");
  const ScalarKind kind = scalarOf(args[0].type());
  if (!std::ranges::all_of(vectorSteps(fn, known), [&](double s) { return !overflows(s, kind); }))
    throw ShaderError(ErrorCode::InvalidConstant, std::format("{}: a step of the constant result overflows {}", name, typeName(scalarType(kind))));
}

Val call(BuiltinFunction fn, TypePtr type, const std::vector<Val>& args) {
  requireConstantDomain(fn, type, args);
  std::vector<ExprPtr> nodes;
  for (const Val& a : args) nodes.push_back(a.node());
  Val result(makeExpr(ex::Call{fn, std::move(nodes)}, std::move(type)));
  // the evaluator covers what the steps above do not, such as every order of a sum
  if (const auto reason = nonFiniteConstant(*result.node()))
    throw ShaderError(ErrorCode::InvalidConstant, std::format("{}: {}", builtinName(fn), *reason));
  return result;
}

/** Checks the first argument against the row of the builtin, then makes the call. */
Val checked(BuiltinFunction fn, const std::vector<Val>& args) {
  return call(fn, checkBuiltin(fn, args.front().type()), args);
}

/** Returns the value, or a constant holding it when it has side effects, so it runs once. */
Val once(const Val& v) { return hasSideEffects(*v.node()) ? Let(v, "once") : v; }

bool isScalarOrVectorOf(const TypePtr& t, bool (*kind)(const TypePtr&)) {
  return isScalarOrVector(t) && kind(t);
}

/** A float scalar or vector, the type that the elementwise float builtins take (not matrices). */
Val floatArg(const ValueLike& v, const TypePtr& like = nullptr) {
  const Val x = toVal(v, like ? like : f32());
  if (!isScalarOrVectorOf(x.type(), isFloat))
    throw ShaderError(ErrorCode::TypeMismatch, std::format("Expected a float scalar or vector, got {}", typeName(x.type())));
  return x;
}

/** Unary builtins; a literal argument becomes an f32. */
Val same(BuiltinFunction fn, const ValueLike& a) { return checked(fn, {toVal(a, f32())}); }

/**
 * Widens a scalar operand to the vector type of `like`. Any other operand must
 * already have that type; toVal() does not change an existing Val.
 */
Val widen(const ValueLike& b, const Val& like, std::string_view what) {
  const Val bv = toVal(b, like.type());
  if (bv.type()->kind == TypeKind::Scalar && like.type()->kind == TypeKind::Vector &&
      bv.type()->scalar == like.type()->scalar)
    return construct(like.type(), {bv});
  checkType(bv, like.type(), what);
  return bv;
}

/** Two operands of one type; a scalar of the same kind on either side widens to the vector. */
Val binarySame(BuiltinFunction fn, const ValueLike& a, const ValueLike& b) {
  Val x = coerceFirst(a, b);
  Val y = toVal(b, x.type());
  const auto widens = [](const Val& s, const Val& v) {
    return s.type()->kind == TypeKind::Scalar && v.type()->kind == TypeKind::Vector &&
           s.type()->scalar == v.type()->scalar;
  };
  if (widens(x, y)) x = construct(y.type(), {x});
  else if (widens(y, x)) y = construct(x.type(), {y});
  checkBuiltin(fn, x.type());
  if (!typesEqual(x.type(), y.type()))
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{}: mismatched {}, {}", builtinName(fn), typeName(x.type()),
                                  typeName(y.type())));
  return call(fn, x.type(), {x, y});
}

} // namespace

Val abs(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Abs, a); }, a);
}
Val sign(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Sign, a); }, a);
}
Val floor(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Floor, a); }, a);
}
Val ceil(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Ceil, a); }, a);
}
Val fract(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Fract, a); }, a);
}
Val round(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Round, a); }, a);
}
Val trunc(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Trunc, a); }, a);
}
Val sqrt(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Sqrt, a); }, a);
}
Val inverseSqrt(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::InverseSqrt, a); }, a);
}
Val exp(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Exp, a); }, a);
}
Val exp2(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Exp2, a); }, a);
}
Val log(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Log, a); }, a);
}
Val log2(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Log2, a); }, a);
}
Val sin(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Sin, a); }, a);
}
Val cos(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Cos, a); }, a);
}
Val tan(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Tan, a); }, a);
}
Val asin(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Asin, a); }, a);
}
Val acos(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Acos, a); }, a);
}
Val atan(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Atan, a); }, a);
}
Val saturate(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Saturate, a); }, a);
}
Val normalize(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Normalize, a); }, a);
}
namespace {

/**
 * Derivatives exist only in the fragment stage. A function body is built
 * outside a stage, so analysis checks the stages that call it.
 */
Val derivative(BuiltinFunction fn, const ValueLike& a) {
  const auto stage = currentStage();
  if (stage && *stage != StageName::Fragment && functionReturnTypes().empty())
    throw ShaderError(ErrorCode::WrongStage, std::format("{} is only valid in the fragment stage", builtinName(fn)));
  return same(fn, a);
}

} // namespace

Val dpdx(const ValueLike& a) {
  return detail::guard([&]() -> Val { return derivative(BuiltinFunction::Dpdx, a); }, a);
}
Val dpdy(const ValueLike& a) {
  return detail::guard([&]() -> Val { return derivative(BuiltinFunction::Dpdy, a); }, a);
}
Val fwidth(const ValueLike& a) {
  return detail::guard([&]() -> Val { return derivative(BuiltinFunction::Fwidth, a); }, a);
}

Val min(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val { return binarySame(BuiltinFunction::Min, a, b); }, a, b);
}
Val max(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val { return binarySame(BuiltinFunction::Max, a, b); }, a, b);
}
Val pow(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val { return binarySame(BuiltinFunction::Pow, a, b); }, a, b);
}
Val atan2(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val {
    return binarySame(BuiltinFunction::Atan2, a, b);
  }, a, b);
}
Val mod(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val {
    const Val result = binarySame(BuiltinFunction::Mod, a, b);
    // a - b * floor(a / b): with both known, a zero divisor component makes a constant inf or NaN,
    // which WGSL rejects; a run time dividend keeps the division at run time, where it is valid
    const auto* c = std::get_if<ex::Call>(&result.node()->node);
    const auto divisor = c ? knownRanges(*c->args[1]) : std::nullopt;
    const auto zero = [](const KnownRange& r) { return mayBe(r, 0); };
    if (divisor && knownRanges(*c->args[0]) && std::ranges::any_of(*divisor, zero))
      throw ShaderError(ErrorCode::InvalidConstant, "mod: the divisor is zero, which makes the constant result infinite or NaN");
    return result;
  }, a, b);
}

Val step(const ValueLike& edge, const ValueLike& x) {
  return detail::guard([&]() -> Val {
    const Val xv = floatArg(x);
    const Val e = toVal(edge, xv.type());
    return call(BuiltinFunction::Step, xv.type(), {widen(e, xv, "step edge"), xv});
  }, edge, x);
}

namespace {

/** WGSL rejects the shader when constant bounds of clamp() are reversed. */
void requireOrderedBounds(const Val& lo, const Val& hi) {
  // the same value twice is equal bounds, whichever value WGSL gives it
  if (sameValue(*lo.node(), *hi.node())) return;
  const auto low = knownRanges(*lo.node());
  const auto high = knownRanges(*hi.node());
  if (!low || !high || low->size() != high->size()) return;
  // reversed for some of the values WGSL may give the bounds
  for (std::size_t i = 0; i < low->size(); ++i)
    if ((*low)[i].high > (*high)[i].low) {
      const bool exact = isExact((*low)[i]) && isExact((*high)[i]);
      throw ShaderError(ErrorCode::InvalidConstant, std::format("clamp: the low bound {} {} above the high bound {}{}", describe((*low)[i]),
                                    exact ? "is" : "can be", describe((*high)[i]),
                                    low->size() > 1 ? std::format(" in component {}", i) : ""));
    }
}

} // namespace

Val clamp(const ValueLike& x, const ValueLike& lo, const ValueLike& hi) {
  return detail::guard([&]() -> Val {
    const Val v = toVal(x);
    const Val low = widen(lo, v, "clamp low bound");
    const Val high = widen(hi, v, "clamp high bound");
    requireOrderedBounds(low, high);
    return checked(BuiltinFunction::Clamp, {v, low, high});
  }, x, lo, hi);
}

Val mix(const ValueLike& a, const ValueLike& b, const ValueLike& t) {
  return detail::guard([&]() -> Val {
    const Val x = coerceFirst(a, b);
    checkBuiltin(BuiltinFunction::Mix, x.type());
    const Val y = toVal(b, x.type());
    checkType(y, x.type(), "mix second value");
    return call(BuiltinFunction::Mix, x.type(), {x, y, widen(t, x, "mix factor")});
  }, a, b, t);
}

Val smoothstep(const ValueLike& e0, const ValueLike& e1, const ValueLike& x) {
  return detail::guard([&]() -> Val {
    const Val xv = floatArg(x);
    const Val low = widen(e0, xv, "smoothstep low edge");
    const Val high = widen(e1, xv, "smoothstep high edge");
    // WGSL rejects equal constant edges; reversed edges are valid there and lowered for GLSL
    if (sameValue(*low.node(), *high.node()) && isConstExpression(*low.node()))
      throw ShaderError(ErrorCode::InvalidConstant, "smoothstep: both edges are the same constant value");
    const auto l = knownRanges(*low.node());
    const auto h = knownRanges(*high.node());
    // equal for some of the values WGSL may give the edges
    const auto overlap = [](const KnownRange& a, const KnownRange& b) { return a.low <= b.high && b.low <= a.high; };
    if (l && h && l->size() == h->size())
      for (std::size_t i = 0; i < l->size(); ++i)
        if (overlap((*l)[i], (*h)[i])) {
          const std::string where = l->size() > 1 ? std::format(" in component {}", i) : "";
          if (isExact((*l)[i]) && isExact((*h)[i]))
            throw ShaderError(ErrorCode::InvalidConstant, std::format("smoothstep: both edges are {}{}", (*l)[i].low, where));
          throw ShaderError(ErrorCode::InvalidConstant, std::format("smoothstep: the edges {} and {} can be equal{}", describe((*l)[i]),
                                        describe((*h)[i]), where));
        }
    return call(BuiltinFunction::Smoothstep, xv.type(), {low, high, xv});
  }, e0, e1, x);
}

Val length(const ValueLike& a) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::Length, {floatArg(a)}); }, a);
}

Val distance(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val {
    const Val x = floatArg(a);
    const Val y = floatArg(b, x.type());
    checkType(y, x.type(), "distance second point");
    return checked(BuiltinFunction::Distance, {x, y});
  }, a, b);
}

Val dot(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val {
    const Val x = toVal(a);
    const Val y = toVal(b, x.type());
    checkType(y, x.type(), "dot second vector");
    return checked(BuiltinFunction::Dot, {x, y});
  }, a, b);
}

Val cross(const Val& a, const Val& b) {
  return detail::guard([&]() -> Val {
    checkType(b, a.type(), "cross second vector");
    return checked(BuiltinFunction::Cross, {a, b});
  }, a, b);
}

Val reflect(const Val& i, const Val& n) {
  return detail::guard([&]() -> Val {
    checkBuiltin(BuiltinFunction::Reflect, i.type());
    checkType(n, i.type(), "reflect normal");
    return checked(BuiltinFunction::Reflect, {i, n});
  }, i, n);
}

Val faceForward(const Val& n, const Val& i, const Val& nref) {
  return detail::guard([&]() -> Val {
    checkBuiltin(BuiltinFunction::FaceForward, n.type());
    checkType(i, n.type(), "faceForward incident");
    checkType(nref, n.type(), "faceForward reference");
    return checked(BuiltinFunction::FaceForward, {n, i, nref});
  }, n, i, nref);
}

Val transpose(const Val& m) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::Transpose, {m}); }, m);
}

Val determinant(const Val& m) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::Determinant, {m}); }, m);
}

Val any(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::Any, {v}); }, v);
}
Val all(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::All, {v}); }, v);
}

Val floatBitsToUint(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::FloatBitsToUint, {v}); }, v);
}
Val uintBitsToFloat(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::UintBitsToFloat, {v}); }, v);
}
Val floatBitsToInt(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::FloatBitsToInt, {v}); }, v);
}
Val intBitsToFloat(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::IntBitsToFloat, {v}); }, v);
}

Val sinh(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Sinh, a); }, a);
}
Val cosh(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Cosh, a); }, a);
}
Val tanh(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Tanh, a); }, a);
}
Val asinh(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Asinh, a); }, a);
}
Val acosh(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Acosh, a); }, a);
}
Val atanh(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Atanh, a); }, a);
}
Val radians(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Radians, a); }, a);
}
Val degrees(const ValueLike& a) {
  return detail::guard([&]() -> Val { return same(BuiltinFunction::Degrees, a); }, a);
}

Val refract(const Val& i, const Val& n, const ValueLike& eta) {
  return detail::guard([&]() -> Val {
    checkBuiltin(BuiltinFunction::Refract, i.type());
    checkType(n, i.type(), "refract normal");
    // the ratio has the component type of the vectors
    const TypePtr& component = scalarType(i.type()->scalar);
    const Val ratio = toVal(eta, component);
    checkType(ratio, component, "refract ratio");
    return checked(BuiltinFunction::Refract, {i, n, ratio});
  }, i, n, eta);
}

Val inverse(const Val& m) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::Inverse, {m}); }, m);
}

Val cbrt(const ValueLike& a) {
  return detail::guard([&]() -> Val {
    // Neither backend has a cube root; pow() is not defined for negative bases.
    const Val x = once(floatArg(a));
    return sign(x) * pow(abs(x), 1.0 / 3.0);
  }, a);
}

Val oneMinus(const ValueLike& a) {
  return detail::guard([&]() -> Val { return binary(BinaryOp::Sub, 1, floatArg(a)); }, a);
}
Val reciprocal(const ValueLike& a) {
  return detail::guard([&]() -> Val { return binary(BinaryOp::Div, 1, floatArg(a)); }, a);
}
Val negate(const Val& a) {
  return detail::guard([&]() -> Val { return -a; }, a);
}

Val pow2(const ValueLike& a) {
  return detail::guard([&]() -> Val {
    const Val x = once(floatArg(a));
    return x * x;
  }, a);
}

Val pow3(const ValueLike& a) {
  return detail::guard([&]() -> Val {
    const Val x = once(floatArg(a));
    return x * x * x;
  }, a);
}

Val pow4(const ValueLike& a) {
  return detail::guard([&]() -> Val {
    const Val x2 = once(pow2(a));
    return x2 * x2;
  }, a);
}

Val lengthSq(const Val& a) {
  return detail::guard([&]() -> Val {
    const Val x = once(a);
    return dot(x, x);
  }, a);
}

Val difference(const ValueLike& a, const ValueLike& b) {
  return detail::guard([&]() -> Val {
    const Val x = coerceFirst(a, b);
    // a u32 subtraction wraps and abs() of a u32 is itself: subtract the smaller from the larger
    if (x.type()->scalar == ScalarKind::U32) {
      const Val p = once(x);
      const Val q = once(toVal(b, p.type()));
      return binary(BinaryOp::Sub, max(p, q), min(p, q));
    }
    return abs(binary(BinaryOp::Sub, x, b));
  }, a, b);
}

Val remap(const ValueLike& x, const ValueLike& inLow, const ValueLike& inHigh,
          const ValueLike& outLow, const ValueLike& outHigh) {
  return detail::guard([&]() -> Val {
    const Val v = floatArg(x);
    const auto f = [&](const ValueLike& p) { return once(toVal(p, v.type())); };
    const Val lo = f(inLow);
    const Val outLo = f(outLow);
    const Val t = (v - lo) / (f(inHigh) - lo);
    return outLo + t * (f(outHigh) - outLo);
  }, x, inLow, inHigh, outLow, outHigh);
}

Val remapClamp(const ValueLike& x, const ValueLike& inLow, const ValueLike& inHigh,
               const ValueLike& outLow, const ValueLike& outHigh) {
  return detail::guard([&]() -> Val {
    const Val v = floatArg(x);
    const auto f = [&](const ValueLike& p) { return once(toVal(p, v.type())); };
    const Val lo = f(inLow);
    const Val outLo = f(outLow);
    const Val t = clamp((v - lo) / (f(inHigh) - lo), 0.0, 1.0);
    return outLo + t * (f(outHigh) - outLo);
  }, x, inLow, inHigh, outLow, outHigh);
}

Val transformDirection(const Val& direction, const Val& matrix) {
  return detail::guard([&]() -> Val {
    checkType(direction, vec3f(), "transformDirection direction");
    checkType(matrix, mat4f(), "transformDirection matrix");
    return normalize((matrix * vec4(direction, 0.0)).xyz());
  }, direction, matrix);
}

Val packHalf2x16(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::PackHalf2x16, {v}); }, v);
}
Val packSnorm2x16(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::PackSnorm2x16, {v}); }, v);
}
Val packUnorm2x16(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::PackUnorm2x16, {v}); }, v);
}
Val unpackHalf2x16(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::UnpackHalf2x16, {v}); }, v);
}
Val unpackSnorm2x16(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::UnpackSnorm2x16, {v}); }, v);
}
Val unpackUnorm2x16(const Val& v) {
  return detail::guard([&]() -> Val { return checked(BuiltinFunction::UnpackUnorm2x16, {v}); }, v);
}

Val select(const ValueLike& cond, const ValueLike& t, const ValueLike& f) {
  return detail::guard([&]() -> Val {
    const Val c = toVal(cond);
    const Val tv = coerceFirst(t, f);
    const Val fv = toVal(f, tv.type());
    if (!typesEqual(tv.type(), fv.type()))
      throw ShaderError(ErrorCode::TypeMismatch, std::format("select branches differ: {}/{}", typeName(tv.type()),
                                    typeName(fv.type())));
    // WGSL select() takes scalars and vectors only
    if (!isScalarOrVector(tv.type()))
      throw ShaderError(ErrorCode::TypeMismatch, std::format("select needs scalar or vector values, got {}; use If", typeName(tv.type())));
    // a bool, or a bool vector that selects each component of vector branches
    const TypePtr& ct = c.type();
    const bool scalarCond = ct == boolType();
    const bool vectorCond = ct->kind == TypeKind::Vector && ct->scalar == ScalarKind::Bool &&
                            tv.type()->kind == TypeKind::Vector && tv.type()->size == ct->size;
    if (!scalarCond && !vectorCond)
      throw ShaderError(ErrorCode::TypeMismatch, std::format("select condition must be bool or a bool vector matching {}, got {}",
                                    typeName(tv.type()), typeName(ct)));
    return Val(makeExpr(ex::Select{c.node(), tv.node(), fv.node()}, tv.type()));
  }, cond, t, f);
}

} // namespace moga
