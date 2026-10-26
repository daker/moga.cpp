#pragma once

// Errors of the analysis passes that keep going after the first one, so a
// build can report every independent problem of a program at once.

#include "moga/Analyze.hpp"
#include "moga/Types.hpp"

#include <optional>
#include <vector>

namespace moga::detail {

using Errors = std::vector<ShaderError>;

/**
 * Analyzes a program, adding the errors of the checking passes to `errors`.
 * Returns nothing when there are some. The later passes (bindings, layouts,
 * names) still throw their first error.
 */
std::optional<AnalyzedProgram> analyzeCollecting(ProgramAST program, const AnalyzeOptions& options, Errors& errors);

} // namespace moga::detail
