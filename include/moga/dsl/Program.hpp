#pragma once

// The stage being built.

#include "moga/dsl/Value.hpp"

#include <optional>

namespace moga {

/** Returns the stage that is being built, or nothing. */
std::optional<StageName> inStage();

} // namespace moga
