#include "Lower.hpp"

#include "Known.hpp"

#include <algorithm>
#include <functional>

namespace moga::detail {

namespace {

bool impure(const std::vector<ExprPtr>& args) {
  return std::ranges::any_of(args, [](const ExprPtr& a) { return hasSideEffects(*a); });
}

/** The inline form repeats operands; with side effects the helper evaluates them once. */
Lowering inlineOrHelper(const std::vector<ExprPtr>& operands, HelperKind kind, const TypePtr& type) {
  if (impure(operands)) return {Form::Helper, moga::Helper{kind, type}};
  return {Form::Inline, std::nullopt};
}

/**
 * GLSL leaves smoothstep undefined when the low edge is not below the high
 * one; WGSL defines it by its formula. Edges known to be in order use the
 * native call; others use a helper with the formula.
 */
Lowering glslSmoothstep(const ex::Call& c, const TypePtr& t) {
  const auto low = knownRanges(*c.args[0]);
  const auto high = knownRanges(*c.args[1]);
  // in order for every value WGSL may give the edges
  const auto below = [](const KnownRange& a, const KnownRange& b) { return a.high < b.low; };
  if (low && high && low->size() == high->size() && std::ranges::equal(*low, *high, below))
    return {};
  return {Form::Helper, moga::Helper{HelperKind::Smoothstep, t}};
}

// GLSL ES 3.00 lacks some overloads that WGSL has; they are written out
Lowering glslCall(const ex::Call& c) {
  if (c.args.empty()) return {};
  const TypePtr& t = c.args[0]->type;
  switch (c.fn) {
  case BuiltinFunction::Saturate:
    return {Form::Inline, std::nullopt};
  case BuiltinFunction::Abs:
    return {t->scalar == ScalarKind::U32 ? Form::Operand : Form::Native, std::nullopt};
  case BuiltinFunction::Any:
  case BuiltinFunction::All:
    return {t->kind == TypeKind::Scalar ? Form::Operand : Form::Native, std::nullopt};
  case BuiltinFunction::Smoothstep:
    return glslSmoothstep(c, t);
  case BuiltinFunction::Dot:
    if (isFloatKind(t->scalar)) return {};
    return inlineOrHelper(c.args, HelperKind::IntegerDot, t);
  default:
    return {};
  }
}

/**
 * A bool selects with ?:, which evaluates one branch; WGSL select() evaluates
 * both, so branches with side effects go through a helper call. A bool vector
 * selects floats with mix() and other types per component.
 */
Lowering glslSelect(const ex::Select& s, const TypePtr& type) {
  if (s.cond->type->kind != TypeKind::Vector) {
    if (impure({s.t, s.f})) return {Form::Helper, moga::Helper{HelperKind::ScalarSelect, type}};
    return {};
  }
  if (isFloatKind(type->scalar)) return {};
  return inlineOrHelper({s.cond, s.t, s.f}, HelperKind::VectorSelect, type);
}

Lowering glsl(const Expr& e) {
  if (const auto* c = std::get_if<ex::Call>(&e.node)) return glslCall(*c);
  if (const auto* s = std::get_if<ex::Select>(&e.node)) return glslSelect(*s, e.type);
  // a matrix record spans four texels; the table declares its own fetchMat4 helper
  if (const auto* t = std::get_if<ex::TableRead>(&e.node); t != nullptr && e.type->kind == TypeKind::Matrix)
    return {hasSideEffects(*t->index) ? Form::Helper : Form::Inline, std::nullopt};
  return {};
}

Lowering wgsl(const Expr& e) {
  if (const auto* c = std::get_if<ex::Call>(&e.node)) {
    // WGSL has no inverse, and its % truncates where mod floors
    if (c->fn == BuiltinFunction::Inverse) return {Form::Helper, moga::Helper{HelperKind::Inverse, e.type}};
    // the inline form repeats both operands, so only a single node (or a vector made from one, as in
    // a broadcast) is written out: nested mods stay linear
    if (c->fn == BuiltinFunction::Mod) {
      const bool small = std::ranges::all_of(c->args, [](const ExprPtr& a) { return a->treeSize <= 2; });
      if (small) return inlineOrHelper(c->args, HelperKind::FloatMod, e.type);
      return {Form::Helper, moga::Helper{HelperKind::FloatMod, e.type}};
    }
    return {};
  }
  const auto* k = std::get_if<ex::Construct>(&e.node);
  if (k == nullptr || e.type->kind != TypeKind::Matrix || k->args.size() != 1) return {};
  // WGSL has neither a mat3 from mat4 constructor nor a diagonal one
  const TypePtr& arg = k->args[0]->type;
  if (e.type->cols == 3 && arg->kind == TypeKind::Matrix && arg->cols == 4)
    return inlineOrHelper(k->args, HelperKind::Mat3FromMat4, e.type);
  if (k->args[0]->type == scalarType(e.type->scalar)) return {Form::Inline, std::nullopt};
  return {};
}

} // namespace

Lowering lower(Target target, const Expr& e) { return target == Target::GLSL ? glsl(e) : wgsl(e); }

} // namespace moga::detail
