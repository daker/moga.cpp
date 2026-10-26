#pragma once

// One call that records, analyzes and compiles a program, and reports every
// problem as a diagnostic instead of throwing (ADR 0004).

#include "moga/Analyze.hpp"
#include "moga/Compile.hpp"
#include "moga/Diagnostic.hpp"
#include "moga/dsl/Compute.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moga {

/** Options of a build: the backends to compile and the analysis options. */
struct BuildOptions {
  /** Compiles GLSL with these options; nothing compiles GLSL when empty. Ignored for compute programs. */
  std::optional<GLSLOptions> glsl = GLSLOptions{};
  /** Compiles WGSL when true. */
  bool wgsl = true;
  /** Options of the usage analysis, for example the resources to retain. */
  AnalyzeOptions analyze;
  /**
   * Records the C++ location of each statement, so generated lines map back
   * to it (the line maps) and errors found after recording can point at it.
   * On by default in builds without NDEBUG.
   */
#ifdef NDEBUG
  bool sourceLocations = false;
#else
  bool sourceLocations = true;
#endif
};

/** The result of a build: the analyzed program, the compiled code and all diagnostics. */
struct [[nodiscard]] BuildResult {
  /** Empty when recording or analysis failed. */
  std::optional<AnalyzedProgram> program;
  /** The GLSL code. Empty when GLSL is not compiled or when the build failed. */
  std::optional<CompiledGLSL> glsl;
  /** The WGSL code. Empty when WGSL is not compiled or when the build failed. */
  std::optional<CompiledWGSL> wgsl;
  /** The errors and warnings of the build, in the order that the build found them. */
  std::vector<Diagnostic> diagnostics;

  /** Returns true when no diagnostic is an error. */
  bool ok() const;

  /**
   * Reads a WebGL shader info log (getShaderInfoLog) for one stage of the
   * GLSL output, and returns its "ERROR: 0:line: ..." and "WARNING: ..."
   * messages, each at the C++ statement that made that line. A line between
   * statements maps to the statement before it; build with source locations,
   * or no message has a C++ location.
   */
  std::vector<Diagnostic> mapGLSLLog(StageName stage, std::string_view log) const;
  /**
   * Same for WGSL: reads ":line:column error|warning|note: ..." messages, as
   * Tint prints them (GPUCompilationInfo text, or the browser console).
   */
  std::vector<Diagnostic> mapWGSLLog(std::string_view log) const;
};

/**
 * Builds a program whose functions, uniforms and other declarations are made
 * before its stages. While it lives, nothing in the DSL throws: every mistake,
 * in a declaration, an expression or a statement, inside or outside a body,
 * becomes a diagnostic of the build, and recording goes on. A declaration is
 * still made, and an expression gives a placeholder value that later code
 * can use without new errors. Statements record their C++ location when
 * BuildOptions::sourceLocations is on:
 *
 *   ShaderBuild build("pbr", key, options);
 *   const Function dGgx = fn("D_GGX", ...);
 *   const BuildResult r = build.finish([&] { ... }, [&] { ... dGgx(...) ... });
 *
 * Use one ShaderBuild per thread at a time; it is not copyable.
 */
class ShaderBuild {
public:
  /**
   * Starts a build. While the build lives, the DSL calls of this thread
   * report their errors to it. `variantKey` names the variant (see variantKey()).
   */
  ShaderBuild(std::string name, std::string variantKey, BuildOptions options = {});
  /** Ends the build. The DSL calls of this thread report errors as they did before the build. */
  ~ShaderBuild();
  ShaderBuild(const ShaderBuild&) = delete;
  ShaderBuild& operator=(const ShaderBuild&) = delete;

  /** Records the stages, then analyzes and compiles. Throws std::logic_error when called twice. */
  [[nodiscard]] BuildResult finish(const std::function<void()>& vertex, const std::function<void()>& fragment);
  /** Same for a compute program; compute is WebGPU only, so only WGSL is compiled. */
  [[nodiscard]] BuildResult finishCompute(const ComputeOptions& compute, const std::function<void()>& body);

private:
  struct State;
  std::unique_ptr<State> state_;
};

/** Builds a render program from a vertex and a fragment body; a ShaderBuild that finishes at once. */
BuildResult buildShader(std::string name, std::string variantKey, const std::function<void()>& vertex,
                        const std::function<void()>& fragment, const BuildOptions& options = {});

/** Builds a compute program. Compute is WebGPU only, so only WGSL is compiled. */
BuildResult buildComputeShader(std::string name, std::string variantKey, const ComputeOptions& compute,
                               const std::function<void()>& body, const BuildOptions& options = {});

} // namespace moga
