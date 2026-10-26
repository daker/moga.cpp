#pragma once

// Values known when the shader is built, for checks that WGSL makes on
// constant expressions.

#include "moga/Nodes.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moga::detail {

/**
 * The values WGSL may give one component of a constant expression: either
 * neighbor of an inexact float result, zero for a subnormal one. low == high
 * when only one value is possible. A bool is 0 or 1.
 */
struct KnownRange {
  double low = 0;
  double high = 0;
};

/**
 * Returns the possible values of each component of a value that is known
 * when the shader is built (see knownComponents() for which values). A check
 * rejects a value when any value in its range is invalid.
 *
 * Ranges are independent: values tied to one uncertain named constant (c and
 * 2 - c) are not known to move together, so a check can reject a valid
 * shader, never accept an invalid one (docs/adr/0006).
 */
std::optional<std::vector<KnownRange>> knownRanges(const Expr& e);

/** Whether the range holds one value. */
inline bool isExact(const KnownRange& r) { return r.low == r.high; }

/** Whether v is one of the possible values. */
inline bool mayBe(const KnownRange& r, double v) { return r.low <= v && v <= r.high; }

/**
 * Whether two expressions always have the same value: the same named
 * constant, or constructors of such values. A named constant is declared
 * once, so its uses hold the same choice among its possible values, which
 * separate ranges cannot show. Any other expression is printed at each use,
 * and each copy may round its own way.
 */
bool sameValue(const Expr& a, const Expr& b);

/** The range for a message: "3", or "2.9999998 to 3.0000002" when it holds more than one value. */
std::string describe(const KnownRange& r);

/**
 * Returns the components of a value that is known when the shader is built,
 * when each has one possible value:
 * a number literal, a named constant, a negation, + - * / of known values
 * or a vector constructor of them.
 * Returns nothing for any other value.
 */
std::optional<std::vector<double>> knownComponents(const Expr& e);

/**
 * Returns true when WGSL evaluates the expression when it creates the
 * shader (a const expression): literals, named constants, operators,
 * constructors, swizzles and calls of constant evaluable builtins. This is
 * a question of form; knownComponents() computes the value of a subset.
 *
 * A bitcast constant, and a constant that reads one, is emitted as a WGSL
 * function and is not a const expression. `allowRuntimeConstants` accepts
 * them, for the value of constant() itself.
 */
bool isConstExpression(const Expr& e, bool allowRuntimeConstants = false);

/**
 * When a float expression is known when the shader is built and a component
 * is infinite or NaN in its type, returns why, for example "the constant
 * value is infinite in f32, which WGSL rejects" (division by zero, overflow
 * of f32 or f16). The caller names the operation in the error.
 */
std::optional<std::string> nonFiniteConstant(const Expr& e);

} // namespace moga::detail
