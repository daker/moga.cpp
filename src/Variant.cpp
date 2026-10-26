#include "moga/Variant.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <format>

namespace moga {

namespace {

/** Escapes the characters that separate the parts of a key. */
std::string escape(std::string_view s) {
  std::string out;
  for (const char c : s) {
    if (c == '\\' || c == '|' || c == '=' || c == '"') out += '\\';
    out += c;
  }
  return out;
}

// Each value shows its type, so true, 1 and "1" give different keys.
struct VariantText {
  std::string operator()(bool v) const { return v ? "true" : "false"; }
  // every NaN prints as nan, so a NaN shows its bits
  std::string operator()(double v) const {
    return std::isnan(v) ? std::format("nan:{:x}", std::bit_cast<std::uint64_t>(v)) : std::format("{}", v);
  }
  std::string operator()(const std::string& v) const { return std::format("\"{}\"", escape(v)); }
};

} // namespace

std::string variantKey(std::string_view base, const VariantOptions& options) {
  std::string key = escape(base);
  for (const auto& [k, v] : options) key += std::format("|{}={}", escape(k), std::visit(VariantText{}, v));
  return key;
}

} // namespace moga
