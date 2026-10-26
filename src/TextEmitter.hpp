#pragma once

#include "moga/Compile.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace moga::detail {

std::string sanitize(std::string_view name);
std::string localName(const LocalVar& v);
/** A constant name: sanitized, and never a builtin function name (constants are not prefixed). */
std::string constantName(std::string_view name);
/** Letters and digits of a type name joined by underscores, for helper names ("float[3]" gives "float_3"). */
std::string typeTag(std::string_view glslOrWgslType);
std::string functionName(const FunctionDecl& f);
std::string structName(std::string_view name);
/** Float literal with a guaranteed decimal point or exponent. */
std::string formatFloat(double v);
std::string join(const std::vector<std::string>& parts, std::string_view sep);

/** Indented line builder that records a line to source mapping. */
class TextEmitter {
public:
  void line(std::string_view text = {}, const std::optional<DebugInfo>& debug = std::nullopt);
  void indent() { ++indent_; }
  void dedent() { --indent_; }
  std::string str() const;
  std::vector<LineMapping> mapping;

private:
  std::vector<std::string> lines_;
  int indent_ = 0;
};

} // namespace moga::detail
