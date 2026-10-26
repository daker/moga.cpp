#pragma once

#include <moga/Types.hpp>

#include <ostream>

namespace moga {
/** Lets doctest print types in failed checks. */
inline std::ostream& operator<<(std::ostream& os, const TypePtr& t) { return os << typeName(t); }
} // namespace moga
