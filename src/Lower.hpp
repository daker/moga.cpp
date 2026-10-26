#pragma once

// Backend lowering decisions (ADR 0005). One function per backend chooses how
// each expression is written; analysis collects the helpers from it and the
// printers write the chosen form without deciding again.

#include "moga/Analyze.hpp"

#include <optional>

namespace moga::detail {

enum class Target { GLSL, WGSL };

/** How a backend writes an expression. What each form means depends on the node. */
enum class Form {
  /** The backend's own construct for the node. */
  Native,
  /** Written out in place from simpler operations. */
  Inline,
  /** A call to a generated helper, so each operand is evaluated once. */
  Helper,
  /** Only the first operand: the operation does nothing for its type. */
  Operand
};

struct Lowering {
  Form form = Form::Native;
  /** The helper function to declare, when the form needs one. */
  std::optional<moga::Helper> helper;
};

/** Returns how the target writes the expression. */
Lowering lower(Target target, const Expr& e);

} // namespace moga::detail
