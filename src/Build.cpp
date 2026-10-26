#include "moga/Build.hpp"
#include "Steps.hpp"

#include "moga/dsl/Program.hpp"
#include "moga/dsl/Value.hpp"

#include "Errors.hpp"
#include "dsl/Context.hpp"

#include <algorithm>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

namespace moga {

namespace {

Diagnostic toDiagnostic(const ShaderError& e, BuildStep step, std::optional<StageName> stage) {
  return Diagnostic{.code = e.code(), .step = step, .stage = stage, .where = e.where(), .message = e.message()};
}

/** Runs one step; a shader error becomes a diagnostic. Returns false when the step failed. */
template <class F> bool attempt(BuildResult& result, BuildStep step, F&& body) {
  try {
    body();
    return true;
  } catch (const ShaderError& e) {
    result.diagnostics.push_back(toDiagnostic(e, step, std::nullopt));
    return false;
  }
}

/** Analyzes and compiles a recorded program. */
void analyzeAndCompile(BuildResult& result, ProgramAST ast, const BuildOptions& options, bool glsl) {
  detail::Errors errors;
  attempt(result, BuildStep::Analyze,
          [&] { result.program = detail::analyzeCollecting(std::move(ast), options.analyze, errors); });
  for (const ShaderError& e : errors) result.diagnostics.push_back(toDiagnostic(e, BuildStep::Analyze, std::nullopt));
  if (!result.program) return;
  // check both targets before emitting, so every gap is reported at once
  const std::vector<Diagnostic> glslGaps = glsl ? checkGLSL(*result.program, *options.glsl) : std::vector<Diagnostic>{};
  const std::vector<Diagnostic> wgslGaps = options.wgsl ? checkWGSL(*result.program) : std::vector<Diagnostic>{};
  result.diagnostics.insert(result.diagnostics.end(), glslGaps.begin(), glslGaps.end());
  result.diagnostics.insert(result.diagnostics.end(), wgslGaps.begin(), wgslGaps.end());
  if (glsl && glslGaps.empty())
    attempt(result, BuildStep::Emit, [&] { result.glsl = compileGLSL(*result.program, *options.glsl); });
  if (options.wgsl && wgslGaps.empty())
    attempt(result, BuildStep::Emit, [&] { result.wgsl = compileWGSL(*result.program); });
}

/** Returns the location of the last statement at or before a generated line. */
std::optional<DebugInfo> sourceOf(const std::vector<LineMapping>& lines, int line) {
  const LineMapping* best = nullptr;
  for (const LineMapping& m : lines)
    if (m.line <= line && (best == nullptr || m.line > best->line)) best = &m;
  if (best == nullptr) return std::nullopt;
  return best->debug;
}

Severity severityOf(std::string_view word) {
  return word == "ERROR" || word == "error" ? Severity::Error : Severity::Warning;
}

/** The capture groups of a log pattern; the message is always the last one. */
struct LogGroups {
  int severity;
  int line;
  int message = 3;
};

/** Finds the messages of a compile log with `pattern` and maps each line to C++. */
std::vector<Diagnostic> mapLog(std::string_view log, const std::regex& pattern, LogGroups groups,
                               const std::vector<LineMapping>& lines, std::optional<StageName> stage) {
  std::vector<Diagnostic> out;
  const std::string text(log);
  for (std::sregex_iterator m(text.begin(), text.end(), pattern), end; m != end; ++m) {
    const int line = std::stoi((*m)[groups.line].str());
    out.push_back({.severity = severityOf((*m)[groups.severity].str()),
                   .code = ErrorCode::Driver,
                   .step = BuildStep::Driver,
                   .stage = stage,
                   .where = sourceOf(lines, line),
                   .generatedLine = line,
                   .message = (*m)[groups.message].str()});
  }
  return out;
}

} // namespace

std::vector<Diagnostic> BuildResult::mapGLSLLog(StageName stage, std::string_view log) const {
  if (!glsl) return {};
  // "ERROR: 0:57: message"; the 0 is the source string index
  static const std::regex pattern(R"((ERROR|WARNING): \d+:(\d+): ([^\r\n]*))");
  return mapLog(log, pattern, {.severity = 1, .line = 2},
                stage == StageName::Vertex ? glsl->vertexLineMap : glsl->fragmentLineMap, stage);
}

std::vector<Diagnostic> BuildResult::mapWGSLLog(std::string_view log) const {
  if (!wgsl) return {};
  // ":57:12 error: message", with an optional file name before the first colon
  static const std::regex pattern(R"(:(\d+):\d+ (error|warning|note): ([^\r\n]*))");
  return mapLog(log, pattern, {.severity = 2, .line = 1}, wgsl->lineMap, std::nullopt);
}

bool BuildResult::ok() const {
  return std::ranges::none_of(diagnostics, [](const Diagnostic& d) { return d.severity == Severity::Error; });
}

/**
 * Body level recovery and source locations for the life of a ShaderBuild: an
 * error ends its body, and recording goes on after it.
 */
struct ShaderBuild::State {
  State(std::string name, std::string variantKey, BuildOptions options)
      : name(std::move(name)), variantKey(std::move(variantKey)), options(std::move(options)),
        locations(this->options.sourceLocations), previous(std::exchange(detail::recordedErrors(), &errors)) {}
  ~State() {
    detail::recordedErrors() = previous;
    detail::discardOrphans();
  }
  State(const State&) = delete;
  State& operator=(const State&) = delete;

  /** Records the program, then analyzes and compiles it when recording had no error. */
  template <class F> BuildResult finish(F&& record, bool glsl) {
    if (finished) throw std::logic_error("ShaderBuild::finish called twice");
    finished = true;
    BuildResult result;
    std::optional<ProgramAST> ast;
    try {
      ast = record();
    } catch (const ShaderError& e) {
      // an error outside any body, such as invalid program options
      errors.push_back({e, std::nullopt});
    }
    for (const detail::RecordedError& e : errors)
      result.diagnostics.push_back(toDiagnostic(e.error, BuildStep::Record, e.stage));
    // an error leaves the program incomplete
    if (errors.empty()) analyzeAndCompile(result, std::move(*ast), options, glsl);
    return result;
  }

  std::string name;
  std::string variantKey;
  BuildOptions options;
  std::vector<detail::RecordedError> errors;
  detail::SourceLocationScope locations;
  std::vector<detail::RecordedError>* previous;
  bool finished = false;
};

ShaderBuild::ShaderBuild(std::string name, std::string variantKey, BuildOptions options)
    : state_(std::make_unique<State>(std::move(name), std::move(variantKey), std::move(options))) {}

ShaderBuild::~ShaderBuild() = default;

BuildResult ShaderBuild::finish(const std::function<void()>& vertex, const std::function<void()>& fragment) {
  State& s = *state_;
  return s.finish([&] { return program(s.name, s.variantKey, vertex, fragment); }, s.options.glsl.has_value());
}

BuildResult ShaderBuild::finishCompute(const ComputeOptions& compute, const std::function<void()>& body) {
  State& s = *state_;
  return s.finish([&] { return computeProgram(s.name, s.variantKey, compute, body); }, false);
}

BuildResult buildShader(std::string name, std::string variantKey, const std::function<void()>& vertex,
                        const std::function<void()>& fragment, const BuildOptions& options) {
  ShaderBuild build(std::move(name), std::move(variantKey), options);
  return build.finish(vertex, fragment);
}

BuildResult buildComputeShader(std::string name, std::string variantKey, const ComputeOptions& compute,
                               const std::function<void()>& body, const BuildOptions& options) {
  ShaderBuild build(std::move(name), std::move(variantKey), options);
  return build.finishCompute(compute, body);
}

} // namespace moga
