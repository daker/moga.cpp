#pragma once

#include <map>
#include <string>
#include <string_view>
#include <variant>

namespace moga {

/** The value of one variant option: a bool, a number or a string. */
using VariantValue = std::variant<bool, double, std::string>;
/** The options of a program variant, by option name. */
using VariantOptions = std::map<std::string, VariantValue, std::less<>>;

/**
 * Returns the key of a program variant, for example
 * `polydata|clip=2|lighting="phong"|tex=true`. The options are sorted by name,
 * so their order does not change the key. Each value shows its type (strings
 * are quoted) and the separators are escaped, so two different option sets
 * never give the same key. Use the key to cache programs on the two backends.
 */
std::string variantKey(std::string_view base, const VariantOptions& options);

} // namespace moga
