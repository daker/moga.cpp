#include "BuiltinTable.hpp"
#include "Context.hpp"
#include "Known.hpp"
#include "Walk.hpp"
#include "moga/dsl/Constants.hpp"
#include "moga/dsl/Constructors.hpp"
#include "moga/dsl/Math.hpp"

#include <format>
#include <limits>
#include <numbers>
#include <unordered_set>

namespace moga {

using namespace detail;

// ------------------------------------------------------------------ constants

namespace {

bool isBitcastFunction(BuiltinFunction fn) { return builtinSpec(fn).bitcast; }

/** Throws when the expression reads something that is not known at compile time. */
void requireConstant(const Expr& e, std::string_view name) {
  // a bitcast constant is valid here; the backend emits it as a function
  if (!isConstExpression(e, true))
    throw ShaderError(ErrorCode::ConstantRequired, std::format("Constant {}: the value must be known at compile time", name));
}

} // namespace

Val constant(std::string name, const ValueLike& value, Loc loc) {
  detail::noteUserLocation(loc);
  return detail::guard([&]() -> Val {
    const Val v = toVal(value);
    requireConstant(*v.node(), name);
    // WGSL evaluates constants at compile time, where a bitcast to inf or NaN is
    // an error; the backend can only move a whole bitcast of a literal to run time
    const auto* top = std::get_if<ex::Call>(&v.node()->node);
    const bool topBitcast = top != nullptr && isBitcastFunction(top->fn);
    if (topBitcast && !std::holds_alternative<ex::Lit>(top->args[0]->node))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("Constant {}: a bitcast must take a literal", name));
    // a reused value is a shared graph: each node is scanned once
    std::unordered_set<const Expr*> scanned;
    const auto nested = [&](const Expr& e, auto&& self) -> bool {
      if (!scanned.insert(&e).second) return false;
      if (const auto* c = std::get_if<ex::Call>(&e.node); c != nullptr && isBitcastFunction(c->fn))
        return true;
      // every operand kind, select included; another named constant is checked on its own
      if (std::holds_alternative<ex::Constant>(e.node)) return false;
      bool found = false;
      forEachOperand(e, [&](const Expr& operand) { found = found || self(operand, self); });
      return found;
    };
    const bool inner = topBitcast ? nested(*top->args[0], nested) : nested(*v.node(), nested);
    if (inner)
      throw ShaderError(ErrorCode::InvalidConstant, std::format("Constant {}: a bitcast must be the whole value", name));
    auto decl =
        std::make_shared<const ConstantDecl>(ConstantDecl{std::move(name), v.type(), v.node()});
    return Val(makeExpr(ex::Constant{decl}, v.type()));
  }, name, value);
}

namespace consts {

const Val& pi() {
  static const Val v = constant("PI", std::numbers::pi);
  return v;
}

const Val& twoPi() {
  static const Val v = constant("TWO_PI", 2.0 * std::numbers::pi);
  return v;
}

const Val& halfPi() {
  static const Val v = constant("HALF_PI", std::numbers::pi / 2.0);
  return v;
}

const Val& epsilon() {
  static const Val v =
      constant("EPSILON", static_cast<double>(std::numeric_limits<float>::epsilon()));
  return v;
}

const Val& infinity() {
  // The bit pattern of +inf; a literal cannot hold a non finite value.
  static const Val v = constant("INFINITY", uintBitsToFloat(toU32(0x7F800000u)));
  return v;
}

} // namespace consts

} // namespace moga
