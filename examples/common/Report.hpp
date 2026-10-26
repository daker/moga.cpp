#pragma once

// Prints a build: its diagnostics, then the generated code in the sections
// that examples/web/viewer.html reads.

#include <moga/Shader.hpp>

#include <cstdio>
#include <print>

/** Prints each diagnostic as "file:line: error[Code]: message". */
inline void printDiagnostics(const moga::BuildResult& r) {
  for (const moga::Diagnostic& d : r.diagnostics) {
    const char* severity = d.severity == moga::Severity::Error ? "error" : "warning";
    if (d.where)
      std::println(stderr, "{}:{}: {}[{}]: {}", d.where->file, d.where->line, severity, moga::errorCodeName(d.code),
                   d.message);
    else
      std::println(stderr, "{}[{}]: {}", severity, moga::errorCodeName(d.code), d.message);
  }
}

/** Prints the diagnostics. Returns false when the build failed. */
inline bool report(const moga::BuildResult& r) {
  printDiagnostics(r);
  return r.ok();
}

/** Prints the generated code. */
inline void printCode(const moga::BuildResult& r) {
  if (r.glsl) {
    std::println("// ---- GLSL vertex\n{}", r.glsl->vertex);
    std::println("// ---- GLSL fragment\n{}", r.glsl->fragment);
  }
  if (r.wgsl) std::println("// ---- WGSL\n{}", r.wgsl->code);
}
