#pragma once

// The steps behind buildShader(): record, analyze, check and compile. They
// are not installed; the public API is buildShader() (ADR 0004). Tests and
// the benchmark use them to reach one step at a time.

#include "moga/Analyze.hpp"
#include "moga/Compile.hpp"
#include "moga/Diagnostic.hpp"
#include "moga/dsl/Compute.hpp"

#include <functional>
#include <string>
#include <vector>

namespace moga {

/** Builds a program. It runs the vertex body, then the fragment body. Throws the first error. */
ProgramAST program(std::string name, std::string variantKey,
                   const std::function<void()>& vertex, const std::function<void()>& fragment);

/**
 * Builds a compute program. The body runs once for each invocation; a
 * workgroup has workgroupSize[0] * [1] * [2] invocations. WebGPU only.
 * `{8, 8, 1}` sets the workgroup size; `{{64, 1, 1}, 32}` also the subgroup size.
 */
ProgramAST computeProgram(std::string name, std::string variantKey, const ComputeOptions& options,
                          const std::function<void()>& body);

/**
 * Finds what each stage uses, checks the stage interface and assigns bindings.
 * Throws on errors. The result keeps its own copy of the AST (a cheap copy:
 * statements are shared), so `analyzeProgram(program(...))` is safe.
 */
AnalyzedProgram analyzeProgram(ProgramAST ast, const AnalyzeOptions& options = {});

/**
 * Returns every feature the program uses that WebGL2 with these options
 * lacks; compileGLSL() throws the first one. A message names the option to
 * set when an extension provides the feature.
 */
std::vector<Diagnostic> checkGLSL(const AnalyzedProgram& program, const GLSLOptions& options);
/** Returns every feature the program uses that WGSL lacks; compileWGSL() throws the first one. */
std::vector<Diagnostic> checkWGSL(const AnalyzedProgram& program);

/** Makes GLSL ES 3.00 vertex and fragment shaders. Throws when a needed extension is off. */
CompiledGLSL compileGLSL(const AnalyzedProgram& program, const GLSLOptions& options = {});
/** Makes one WGSL module with the vs_main and fs_main entry points. Throws for GLSL only builtins. */
CompiledWGSL compileWGSL(const AnalyzedProgram& program);

} // namespace moga
