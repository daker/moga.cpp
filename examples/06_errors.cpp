// Error management: buildShader() and ShaderBuild report every mistake as a
// diagnostic with a code, the build step, the stage and the C++ line, instead
// of throwing, and one build reports all of them. With no build open, the DSL
// throws moga::ShaderError. Source locations also map driver errors back to C++.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <format>
#include <functional>
#include <print>
#include <string>
#include <string_view>

using namespace moga;

namespace {

std::string_view stepName(BuildStep step) {
  switch (step) {
  case BuildStep::Record:
    return "record";
  case BuildStep::Analyze:
    return "analyze";
  case BuildStep::TargetCheck:
    return "target check";
  case BuildStep::Emit:
    return "emit";
  case BuildStep::Driver:
    return "driver";
  }
  return "?";
}

/** Prints the diagnostics of one build, or that it passed. */
void show(std::string_view what, const BuildResult& r) {
  std::println("{}:{}", what, r.diagnostics.empty() ? " no error" : "");
  for (const Diagnostic& d : r.diagnostics)
    std::println("    [{}]{}: {}[{}]: {}", stepName(d.step), d.where ? std::format(" line {}", d.where->line) : "",
                 d.severity == Severity::Error ? "error" : "warning", errorCodeName(d.code), d.message);
}

/** Runs a declaration made outside buildShader() and prints the error it throws. */
void expectThrow(std::string_view what, const std::function<void()>& body) {
  try {
    body();
    std::println("{}: no error", what);
  } catch (const ShaderError& e) {
    std::println("{}: throws\n    {}[{}]: {}", what, "error", errorCodeName(e.code()), e.what());
  }
}

/** A trivial fragment stage. */
void solidColor(const Output& out) { out.set(vec4(1.0)); }

} // namespace

int main() {
  const Output color = output(0);
  const Val position = attribute("position", vec3f());
  const BuildOptions located{.sourceLocations = true};

  std::println("// ---- errors while recording");
  show("mismatched vector sizes",
       buildShader("bad", "BAD", [&] { Position(vec4(position + vec2(1.0), 1.0)); }, [&] { solidColor(color); },
                   located));
  show("fraction used as an integer", buildShader("bad", "BAD", [&] {
         const Val i = Var(toI32(1), "i");
         i.set(i + 0.5);
         Position(vec4(0.0));
       }, [&] { solidColor(color); }, located));
  show("assignment to a constant", buildShader("bad", "BAD", [&] {
         const Val t = Let(1.0, "t");
         t.set(2.0);
         Position(vec4(0.0));
       }, [&] { solidColor(color); }, located));
  const UniformBlock camera = uniformBlock("camera", {{"viewProj", mat4f()}});
  show("unknown uniform field",
       buildShader("bad", "BAD", [&] { Position(camera["viewproj"] * vec4(position, 1.0)); },
                   [&] { solidColor(color); }, located));
  // one build, three mistakes in three bodies: all three are reported
  show("three mistakes at once", buildShader("bad", "BAD", [&] {
         Position(Var(mat4f()) * position);
       }, [&] {
         If(true, [&] { Discard(); color.set(vec4(position.xy() + vec3(1.0), 1.0)); });
         color.set(vec4(1.0, 2.0));
       }, located));

  std::println("\n// ---- errors when the program is analyzed");
  const Varying normal = varying("normal", vec3f());
  show("varying read but never written",
       buildShader("bad", "BAD", [&] { Position(vec4(position, 1.0)); },
                   [&] { color.set(vec4(normal.value(), 1.0)); }, located));
  show("attribute read in a function", buildShader("bad", "BAD", [&] {
         const Function leak = fn("leak", {}, vec3f(), [&](const Args&) { Return(position); });
         Position(vec4(leak(), 1.0));
       }, [&] { solidColor(color); }, located));

  std::println("\n// ---- features a target lacks, all listed before compiling");
  const auto multiDraw = [&] {
    ClipDistance(0, 1.0);
    Position(vec4(toF32(drawIndex())));
  };
  show("drawIndex and ClipDistance on WebGL2 without extensions",
       buildShader("multi", "MULTI", multiDraw, [&] { solidColor(color); }, {.wgsl = false}));
  show("with WEBGL_multi_draw and WEBGL_clip_cull_distance",
       buildShader("multi", "MULTI", multiDraw, [&] { solidColor(color); },
                   {.glsl = GLSLOptions{.clipDistanceExtension = true, .multiDraw = true}, .wgsl = false}));
  show("drawIndex on WebGPU", buildShader("multi", "MULTI", multiDraw, [&] { solidColor(color); },
                                          {.glsl = std::nullopt}));

  // A ShaderBuild opened before the declarations reports their mistakes too,
  // and a value an error could not make causes no further errors.
  std::println("\n// ---- mistakes before the stages, in a ShaderBuild");
  {
    ShaderBuild build("early", "EARLY", located);
    const Varying id = varying("id", u32());
    const Val scale = vec2(1.0) + vec3(1.0);
    const Function f = fn("f", {{"x", f32()}, {"x", f32()}}, f32(), [](const Args& a) { Return(a["x"]); });
    show("three declarations",
         build.finish([&] { id.set(1u); Position(vec4(scale, 1.0)); }, [&] { color.set(vec4(f(1.0, 2.0))); }));
  }

  std::println("\n// ---- with no build open, the DSL throws");
  expectThrow("integer varying without flat interpolation", [&] { varying("id", u32()); });
  expectThrow("statement outside a stage", [&] { Var(1.0); });

  // With source locations on, the build records the C++ line of each
  // statement, so a driver error on a generated line can be traced back.
  std::println("\n// ---- debug line mapping");
  Stopwatch sw;
  const BuildResult traced = buildShader(
      "traced", "TRACED", [&] { Position(vec4(position, 1.0)); },
      [&] {
        const Val c = Let(sin(fragCoord().x() * 0.1), "c");
        color.set(vec4(c, c, c, 1.0));
      },
      {.glsl = std::nullopt, .sourceLocations = true});
  sw.lap("build");
  if (!report(traced)) return 1;
  for (const LineMapping& m : traced.wgsl->lineMap)
    std::println("WGSL line {} comes from {}:{}", m.line, m.debug.file, m.debug.line);
  sw.print("TRACED");

  // The log here is written by hand, as Tint prints one, for the last mapped line.
  const int lastLine = traced.wgsl->lineMap.back().line;
  const std::string log = std::format(":{}:3 error: an example driver message", lastLine);
  for (const Diagnostic& d : traced.mapWGSLLog(log))
    std::println("{}:{}: driver error on WGSL line {}: {}", d.where->file, d.where->line, *d.generatedLine,
                 d.message);
}
