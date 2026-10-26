#pragma once

// Control flow behaviors, as in WGSL: the set of ways a path can leave a
// statement or a block. Every control flow check is a query on it (ADR 0005).

#include "moga/Nodes.hpp"

namespace moga::detail {

enum Behavior : unsigned {
  /** The path reaches the next statement. */
  kNext = 1,
  /** The path leaves the enclosing loop or switch with Break(). */
  kBreak = 2,
  /** The path starts the next iteration with Continue(). */
  kContinue = 4,
  /** The path returns from the function or stage. */
  kReturn = 8
};

/** A set of Behavior bits. */
using Behaviors = unsigned;

/**
 * A statement that ends a path as satisfied, or null. A satisfied path drops
 * out of the set, so the result holds the ways out that never met the marker
 * (for example, the paths that never write Position()).
 */
using Marker = bool (*)(const Stmt&);

Behaviors blockBehaviors(const Block& block, Marker marker = nullptr);
Behaviors statementBehaviors(const Stmt& s, Marker marker = nullptr);

} // namespace moga::detail
