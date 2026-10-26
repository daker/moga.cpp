#pragma once

// A problem found while building a program, as data (ADR 0004).

#include "moga/Nodes.hpp"
#include "moga/Types.hpp"

#include <optional>
#include <string>

namespace moga {

/** How serious a diagnostic is. An error makes BuildResult::ok() false. */
enum class Severity { Error, Warning };

/** The step of a build that found a problem. */
enum class BuildStep {
  /** The DSL calls that record the program. */
  Record,
  /** The usage analysis of the whole program. */
  Analyze,
  /** The check that each backend supports the features of the program. */
  TargetCheck,
  /** The code generation of a backend. */
  Emit,
  /** Read from the compile log of a GPU driver, after the build. */
  Driver
};

/** One problem that a build found, with its kind, step and location. */
struct Diagnostic {
  /** An error or a warning. */
  Severity severity = Severity::Error;
  /** The kind of problem. Match on it in tests and tools. */
  ErrorCode code = ErrorCode::Internal;
  /** The step of the build that found the problem. */
  BuildStep step = BuildStep::Record;
  /** The stage whose body was being recorded, when the problem is in one. */
  std::optional<StageName> stage;
  /** The C++ location of the code that caused the problem, when known. */
  std::optional<DebugInfo> where;
  /** The 1 based line in the generated code, for a problem a driver reported. */
  std::optional<int> generatedLine;
  /** The message without the location. */
  std::string message;
};

} // namespace moga
