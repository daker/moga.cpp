#pragma once

// Locals are only visible in the block that declares them.

#include "moga/Nodes.hpp"

#include <vector>

namespace moga::detail {

using Errors = std::vector<ShaderError>;

/**
 * Adds an error when a statement reads or writes a local outside the block that
 * declares it: a Val kept in C++ after the If, loop or switch body that made
 * it. Both languages limit a local to its enclosing block. Calls are followed
 * into functions, whose parameters are visible in their whole body.
 */
void checkLocalScopes(const ProgramAST& ast, Errors& errors);

} // namespace moga::detail
