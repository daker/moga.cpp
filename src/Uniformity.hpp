#pragma once

// WGSL uniformity: barriers and derivatives must be reached by every
// invocation of a workgroup (or every fragment of a quad) together.

#include "moga/Nodes.hpp"

#include <vector>

namespace moga::detail {

using Errors = std::vector<ShaderError>;

/**
 * Adds an error when a barrier or a derivative can sit in control flow that differs
 * between invocations: under a condition, loop bound or switch selector that
 * is not uniform, after a return that not every invocation takes, or in a
 * loop after a break or continue that not every invocation takes. Calls are
 * followed into functions.
 *
 * Uniform values are literals, constants, uniform buffers, workgroupId,
 * numWorkgroups, read only storage, tables and textures at uniform
 * coordinates, and builtins and functions of uniform values. A local that is
 * ever assigned a non uniform value, or assigned in non uniform control flow,
 * counts as non uniform everywhere: simpler than WGSL, never less strict.
 */
void checkUniformity(const ProgramAST& ast, Errors& errors);

} // namespace moga::detail
