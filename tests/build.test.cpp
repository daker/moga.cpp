#include <doctest/doctest.h>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace moga;

namespace {
/** A fragment body that writes a constant color. */
void solid(const Output& out) { out.set(vec4(1.0)); }
} // namespace

TEST_CASE("buildShader compiles a valid program for both targets") {
  const Val position = attribute("position", vec3f());
  const Output out = output(0);
  const BuildResult r = buildShader("tri", "TRI", [&] { Position(vec4(position, 1.0)); },
                                    [&] { solid(out); });
  CHECK(r.ok());
  CHECK(r.diagnostics.empty());
  REQUIRE(r.program);
  CHECK(r.program->reflection.attributes.size() == 1);
  REQUIRE(r.glsl);
  CHECK(r.glsl->vertex.starts_with("#version 300 es"));
  REQUIRE(r.wgsl);
  CHECK(r.wgsl->code.find("@vertex") != std::string::npos);
}

TEST_CASE("buildShader compiles only the targets asked for") {
  const Output out = output(0);
  const BuildResult r = buildShader("t", "T", [] { Position(vec4(0.0)); }, [&] { solid(out); },
                                    {.glsl = std::nullopt});
  CHECK(r.ok());
  CHECK_FALSE(r.glsl);
  CHECK(r.wgsl);
}

TEST_CASE("an error while recording becomes a diagnostic with stage and location") {
  const Val position = attribute("position", vec3f());
  const Output out = output(0);
  std::uint32_t line = 0;
  const BuildResult r = buildShader("bad", "BAD", [] { Position(vec4(0.0)); }, [&] {
    line = __LINE__ + 1;
    out.set(vec4(position.xy() + vec3(1.0), 1.0));
  });
  CHECK_FALSE(r.ok());
  CHECK_FALSE(r.program);
  CHECK_FALSE(r.glsl);
  CHECK_FALSE(r.wgsl);
  REQUIRE(r.diagnostics.size() == 1);
  const Diagnostic& d = r.diagnostics[0];
  CHECK(d.severity == Severity::Error);
  CHECK(d.code == ErrorCode::InvalidOperands);
  CHECK(d.step == BuildStep::Record);
  CHECK(d.stage == StageName::Fragment);
  REQUIRE(d.where);
  CHECK(d.where->line == line);
  CHECK(d.where->file.ends_with("build.test.cpp"));
  CHECK_FALSE(d.message.empty());
}

TEST_CASE("an error in the vertex body names the vertex stage") {
  const Output out = output(0);
  const BuildResult r = buildShader("bad", "BAD", [] { Discard(); }, [&] { solid(out); });
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].stage == StageName::Vertex);
  CHECK(r.diagnostics[0].code == ErrorCode::WrongStage);
}

TEST_CASE("an analysis error is reported at the analyze step") {
  const Varying normal = varying("normal", vec3f());
  const Output out = output(0);
  const BuildResult r = buildShader("bad", "BAD", [] { Position(vec4(0.0)); },
                                    [&] { out.set(vec4(normal.value(), 1.0)); });
  CHECK_FALSE(r.ok());
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].step == BuildStep::Analyze);
  CHECK(r.diagnostics[0].code == ErrorCode::StageInterface);
  CHECK_FALSE(r.diagnostics[0].stage);
}

TEST_CASE("a target error is reported at the target check step and keeps the other outputs") {
  const Output out = output(0);
  const auto vertex = [] { Position(vec4(toF32(drawIndex()))); };
  const BuildResult r = buildShader("d", "D", vertex, [&] { solid(out); }, {.wgsl = false});
  CHECK_FALSE(r.ok());
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].step == BuildStep::TargetCheck);
  CHECK(r.diagnostics[0].code == ErrorCode::MissingFeature);
  CHECK(r.program);
  CHECK_FALSE(r.glsl);

  const BuildResult fixed =
      buildShader("d", "D", vertex, [&] { solid(out); }, {.glsl = GLSLOptions{.multiDraw = true}, .wgsl = false});
  CHECK(fixed.ok());
  CHECK(fixed.glsl);
}

TEST_CASE("source locations fill the line maps only when asked for") {
  const Output out = output(0);
  const std::uint32_t line = __LINE__ + 2;
  const BuildResult on = buildShader("dbg", "DBG", [] { Position(vec4(0.0)); },
                                     [&] { out.set(vec4(1.0)); }, {.sourceLocations = true});
  REQUIRE(on.glsl);
  REQUIRE(on.glsl->fragmentLineMap.size() == 1);
  CHECK(on.glsl->fragmentLineMap[0].debug.line == line);

  const BuildResult off = buildShader("dbg", "DBG", [] { Position(vec4(0.0)); },
                                      [&] { out.set(vec4(1.0)); }, {.sourceLocations = false});
  REQUIRE(off.glsl);
  CHECK(off.glsl->fragmentLineMap.empty());
}

TEST_CASE("a failed build does not leave source locations on") {
  const Output out = output(0);
  const BuildResult failed =
      buildShader("bad", "BAD", [] { Discard(); }, [&] { solid(out); }, {.sourceLocations = true});
  CHECK_FALSE(failed.ok());
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { solid(out); });
  CHECK(compileGLSL(analyzeProgram(ast)).fragmentLineMap.empty());
}

TEST_CASE("buildComputeShader compiles WGSL only") {
  const BuildResult r = buildComputeShader("c", "C", {.workgroupSize = {8, 1, 1}}, [] {});
  CHECK(r.ok());
  CHECK_FALSE(r.glsl);
  REQUIRE(r.wgsl);
  CHECK(r.wgsl->code.find("@compute") != std::string::npos);
}

TEST_CASE("the target check reports every missing GLSL extension at once") {
  const Output out = output(0);
  const auto vertex = [] {
    ClipDistance(0, 1.0);
    Position(vec4(toF32(drawIndex())));
  };
  const BuildResult r = buildShader("d", "D", vertex, [&] { solid(out); }, {.wgsl = false});
  CHECK_FALSE(r.glsl);
  REQUIRE(r.diagnostics.size() == 2);
  for (const Diagnostic& d : r.diagnostics) {
    CHECK(d.step == BuildStep::TargetCheck);
    CHECK(d.code == ErrorCode::MissingFeature);
  }
  const auto mentions = [&](std::string_view extension) {
    return std::ranges::any_of(r.diagnostics, [&](const Diagnostic& d) { return d.message.contains(extension); });
  };
  CHECK(mentions("WEBGL_clip_cull_distance"));
  CHECK(mentions("WEBGL_multi_draw"));

  const BuildResult fixed = buildShader("d", "D", vertex, [&] { solid(out); },
                                        {.glsl = GLSLOptions{.clipDistanceExtension = true, .multiDraw = true},
                                         .wgsl = false});
  CHECK(fixed.ok());
}

TEST_CASE("the target check covers both targets and lets the supported one compile") {
  const Output out = output(0);
  const BuildResult r =
      buildShader("d", "D", [] { Position(vec4(toF32(drawIndex()))); }, [&] { solid(out); });
  REQUIRE(r.diagnostics.size() == 2);
  CHECK(r.diagnostics[0].message.contains("WEBGL_multi_draw"));
  CHECK(r.diagnostics[1].message.contains("drawIndex"));
  CHECK_FALSE(r.glsl);
  CHECK_FALSE(r.wgsl);

  const BuildResult glslOnly = buildShader("d", "D", [] { Position(vec4(toF32(drawIndex()))); },
                                           [&] { solid(out); }, {.glsl = GLSLOptions{.multiDraw = true}});
  REQUIRE(glslOnly.diagnostics.size() == 1);
  CHECK(glslOnly.glsl);
  CHECK_FALSE(glslOnly.wgsl);
}

TEST_CASE("checkGLSL lists the WebGPU only features of a program") {
  const BuildResult compute = buildComputeShader("c", "C", {}, [] {});
  REQUIRE(compute.program);
  const std::vector<Diagnostic> gaps = checkGLSL(*compute.program, {});
  REQUIRE_FALSE(gaps.empty());
  CHECK(gaps[0].code == ErrorCode::MissingFeature);
  CHECK(checkWGSL(*compute.program).empty());
}

TEST_CASE("analysis reports every independent error of a program") {
  const Varying normal = varying("normal", vec3f());
  const Output out = output(0);
  std::uint32_t scopeLine = 0;
  std::optional<Val> kept;
  const BuildResult r = buildShader(
      "bad", "BAD", [] { If(true, [] { Position(vec4(0.0)); }); },
      [&] {
        If(true, [&] { kept = Var(1.0, "inner"); });
        scopeLine = __LINE__ + 1;
        out.set(vec4(normal.value() * *kept, 1.0));
      },
      {.sourceLocations = true});
  CHECK_FALSE(r.program);
  REQUIRE(r.diagnostics.size() == 3);
  const auto count = [&](ErrorCode code) {
    return std::ranges::count_if(r.diagnostics, [&](const Diagnostic& d) { return d.code == code; });
  };
  // the unwritten varying, and the Position() missing on the else path
  CHECK(count(ErrorCode::StageInterface) == 2);
  CHECK(count(ErrorCode::Scope) == 1);
  for (const Diagnostic& d : r.diagnostics) {
    CHECK(d.step == BuildStep::Analyze);
    if (d.code != ErrorCode::Scope) continue;
    REQUIRE(d.where);
    CHECK(d.where->line == scopeLine);
  }
}

TEST_CASE("an analysis error inside a function points at its statement") {
  const Val position = attribute("position", vec3f());
  const Output out = output(0);
  std::uint32_t line = 0;
  const BuildResult r = buildShader(
      "bad", "BAD",
      [&] {
        const Function leak = fn("leak", {}, vec3f(), [&](const Args&) {
          line = __LINE__ + 1;
          Return(position);
        });
        Position(vec4(leak(), 1.0));
      },
      [&] { solid(out); }, {.sourceLocations = true});
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].code == ErrorCode::WrongStage);
  REQUIRE(r.diagnostics[0].where);
  CHECK(r.diagnostics[0].where->line == line);
}

TEST_CASE("analyzeProgram still throws the first error") {
  const Varying normal = varying("normal", vec3f());
  const Output out = output(0);
  const ProgramAST ast = program("bad", "BAD", [] {}, [&] { out.set(vec4(normal.value(), 1.0)); });
  try {
    (void)analyzeProgram(ast);
    FAIL("expected an error");
  } catch (const ShaderError& e) {
    CHECK(e.code() == ErrorCode::StageInterface);
  }
}

TEST_CASE("recording goes on after an error in each stage") {
  const Output out = output(0);
  const BuildResult r = buildShader(
      "bad", "BAD", [] { Position(vec4(vec2(1.0) + vec3(1.0), 1.0)); },
      [&] { out.set(vec4(vec3(1.0) * vec2(1.0), 1.0)); });
  REQUIRE(r.diagnostics.size() == 2);
  CHECK(r.diagnostics[0].stage == StageName::Vertex);
  CHECK(r.diagnostics[1].stage == StageName::Fragment);
  CHECK_FALSE(r.program);
}

TEST_CASE("every independent error of a body is reported") {
  const Output out = output(0);
  std::uint32_t first = 0;
  std::uint32_t second = 0;
  std::uint32_t after = 0;
  const BuildResult r = buildShader("bad", "BAD", [] { Position(vec4(0.0)); }, [&] {
    If(true, [&] {
      first = __LINE__ + 1;
      (void)(vec2(1.0) + vec3(1.0));
      second = __LINE__ + 1;
      (void)(vec2(1.0) - vec3(1.0));
    });
    after = __LINE__ + 1;
    out.set(vec4(vec2(1.0) * vec3(1.0), 1.0));
  });
  REQUIRE(r.diagnostics.size() == 3);
  const std::array lines{first, second, after};
  for (std::size_t i = 0; i < lines.size(); ++i) {
    REQUIRE(r.diagnostics[i].where);
    CHECK(r.diagnostics[i].where->line == lines[i]);
  }
}

TEST_CASE("a value an error could not make causes no further errors") {
  const Output out = output(0);
  const BuildResult r = buildShader("bad", "BAD", [] { Position(vec4(0.0)); }, [&] {
    const Val bad = Let(vec2(1.0) + vec3(1.0), "bad");
    // each of these would be an error on its own, but they all use `bad`
    out.set(vec4(bad.xyz() * 2.0, bad.w()));
    bad.set(vec4(1.0));
  });
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].code == ErrorCode::InvalidOperands);
}

TEST_CASE("ShaderBuild reports errors outside any body without throwing") {
  const Output out = output(0);
  std::uint32_t exprLine = 0;
  std::uint32_t declLine = 0;
  std::uint32_t fnLine = 0;
  ShaderBuild build("s", "S");
  exprLine = __LINE__ + 1;
  const Val k = vec2(1.0) + vec3(1.0);
  declLine = __LINE__ + 1;
  const Varying id = varying("id", u32());
  fnLine = __LINE__ + 1;
  const Function f = fn("f", {{"x", f32()}, {"x", f32()}}, f32(), [](const Args& a) { Return(a["x"]); });
  const BuildResult r = build.finish(
      [&] {
        id.set(1u);
        Position(vec4(k, 1.0, 1.0));
      },
      [&] { out.set(vec4(f(1.0, 2.0))); });
  REQUIRE(r.diagnostics.size() == 3);
  CHECK(r.diagnostics[0].code == ErrorCode::InvalidOperands);
  CHECK(r.diagnostics[1].code == ErrorCode::InvalidDeclaration);
  CHECK(r.diagnostics[2].code == ErrorCode::DuplicateName);
  const std::array lines{exprLine, declLine, fnLine};
  for (std::size_t i = 0; i < lines.size(); ++i) {
    REQUIRE(r.diagnostics[i].where);
    CHECK(r.diagnostics[i].where->line == lines[i]);
  }
}

TEST_CASE("a function body cut short by an error reports no missing return") {
  const Output out = output(0);
  const BuildResult r = buildShader("bad", "BAD", [] { Position(vec4(0.0)); }, [&] {
    const Function f = fn("f", {}, f32(), [](const Args&) {
      (void)(vec2(1.0) + vec3(1.0));
      Return(1.0);
    });
    out.set(vec4(f()));
  });
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].code == ErrorCode::InvalidOperands);
}

TEST_CASE("program() without buildShader still throws at the first error") {
  CHECK_THROWS_AS(program("bad", "BAD", [] { If(true, [] { (void)(vec2(1.0) + vec3(1.0)); }); }, [] {}),
                  ShaderError);
}

namespace {
/** Returns the 1 based line of the first line of `code` that contains `text`, or 0. */
int lineOf(std::string_view code, std::string_view text) {
  int line = 1;
  for (std::size_t start = 0; start < code.size(); ++line) {
    const std::size_t end = std::min(code.find('\n', start), code.size());
    if (code.substr(start, end - start).contains(text)) return line;
    start = end + 1;
  }
  return 0;
}
} // namespace

TEST_CASE("a WebGL compile log maps back to the C++ line of the statement") {
  const Output out = output(0);
  const std::uint32_t line = __LINE__ + 2;
  const BuildResult r = buildShader("m", "M", [] { Position(vec4(0.0)); },
                                    [&] { out.set(vec4(0.25)); }, {.wgsl = false, .sourceLocations = true});
  REQUIRE(r.glsl);
  const int generated = lineOf(r.glsl->fragment, "vec4(0.25)");
  REQUIRE(generated > 0);
  const std::string log = std::format("ERROR: 0:{}: 'o_0' : some driver complaint\nnot a message line\n", generated);
  const std::vector<Diagnostic> mapped = r.mapGLSLLog(StageName::Fragment, log);
  REQUIRE(mapped.size() == 1);
  CHECK(mapped[0].code == ErrorCode::Driver);
  CHECK(mapped[0].severity == Severity::Error);
  CHECK(mapped[0].stage == StageName::Fragment);
  CHECK(mapped[0].generatedLine == generated);
  CHECK(mapped[0].message == "'o_0' : some driver complaint");
  REQUIRE(mapped[0].where);
  CHECK(mapped[0].where->line == line);
}

TEST_CASE("a WGSL compile log maps back, and a warning stays a warning") {
  const Output out = output(0);
  const std::uint32_t line = __LINE__ + 2;
  const BuildResult r = buildShader("m", "M", [] { Position(vec4(0.0)); },
                                    [&] { out.set(vec4(0.25)); }, {.glsl = std::nullopt, .sourceLocations = true});
  REQUIRE(r.wgsl);
  const int generated = lineOf(r.wgsl->code, "vec4f(0.25f)");
  REQUIRE(generated > 0);
  const std::string log = std::format("Error while parsing WGSL: :{}:7 error: bad thing\n:{}:3 warning: odd thing\n",
                                      generated, generated);
  const std::vector<Diagnostic> mapped = r.mapWGSLLog(log);
  REQUIRE(mapped.size() == 2);
  CHECK(mapped[0].severity == Severity::Error);
  CHECK(mapped[0].message == "bad thing");
  CHECK(mapped[1].severity == Severity::Warning);
  for (const Diagnostic& d : mapped) {
    REQUIRE(d.where);
    CHECK(d.where->line == line);
  }
}

TEST_CASE("a log line before any statement has no C++ location") {
  const Output out = output(0);
  const BuildResult r = buildShader("m", "M", [] { Position(vec4(0.0)); }, [&] { solid(out); },
                                    {.wgsl = false, .sourceLocations = true});
  const std::vector<Diagnostic> mapped = r.mapGLSLLog(StageName::Fragment, "ERROR: 0:1: '' : version\n");
  REQUIRE(mapped.size() == 1);
  CHECK_FALSE(mapped[0].where);
}

TEST_CASE("ShaderBuild reports errors in functions declared before the stages") {
  const Output out = output(0);
  std::uint32_t line = 0;
  ShaderBuild build("s", "S", {.sourceLocations = true});
  const Function f = fn("f", {}, f32(), [&](const Args&) {
    line = __LINE__ + 1;
    (void)(vec2(1.0) + vec3(1.0));
    Return(1.0);
  });
  const BuildResult r = build.finish([] { Position(vec4(0.0)); }, [&] { out.set(vec4(f())); });
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].code == ErrorCode::InvalidOperands);
  CHECK(r.diagnostics[0].step == BuildStep::Record);
  CHECK_FALSE(r.diagnostics[0].stage);
  REQUIRE(r.diagnostics[0].where);
  CHECK(r.diagnostics[0].where->line == line);
  CHECK_FALSE(r.program);
}

TEST_CASE("ShaderBuild builds a valid program with shared functions") {
  const Output out = output(0);
  ShaderBuild build("s", "S");
  const Function half = fn("half", {{"x", f32()}}, f32(), [](const Args& a) { Return(a["x"] * 0.5); });
  const BuildResult r = build.finish([&] { Position(vec4(half(1.0))); }, [&] { out.set(vec4(half(2.0))); });
  CHECK(r.ok());
  REQUIRE(r.program);
  // one function, shared by both stages
  REQUIRE(r.program->functions.size() == 1);
  CHECK(r.program->functions[0]->name == "half");
}

TEST_CASE("ShaderBuild finishes once") {
  const Output out = output(0);
  ShaderBuild build("s", "S");
  const BuildResult r = build.finish([] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); });
  CHECK(r.ok());
  CHECK_THROWS_AS((void)build.finish([] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); }), std::logic_error);
}

TEST_CASE("ShaderBuild restores error handling when it ends") {
  {
    ShaderBuild build("s", "S", {.sourceLocations = true});
  }
  // errors propagate again, and statements carry no location
  CHECK_THROWS_AS(fn("g", {}, f32(), [](const Args&) {
                    (void)(vec2(1.0) + vec3(1.0));
                    Return(1.0);
                  }),
                  ShaderError);
  const Output out = output(0);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); });
  CHECK(compileGLSL(analyzeProgram(ast)).fragmentLineMap.empty());
}

TEST_CASE("a constructor error points at the constructor call") {
  const Output out = output(0);
  std::uint32_t line = 0;
  const BuildResult r = buildShader("bad", "BAD", [] { Position(vec4(0.0)); }, [&] {
    line = __LINE__ + 1;
    out.set(vec4(1.0, 2.0));
  });
  REQUIRE(r.diagnostics.size() == 1);
  CHECK(r.diagnostics[0].code == ErrorCode::ArgumentCount);
  REQUIRE(r.diagnostics[0].where);
  CHECK(r.diagnostics[0].where->line == line);
}

TEST_CASE("no mistake outside a body escapes a ShaderBuild") {
  const Val v3 = attribute("v3", vec3f());
  const std::vector<std::pair<std::string_view, std::function<void()>>> mistakes{
      {"operands", [] { (void)(vec2(1.0) + vec3(1.0)); }},
      {"constructor", [] { (void)vec4(1.0, 2.0); }},
      {"swizzle", [&] { (void)v3.w(); }},
      {"member", [&] { (void)v3.member("x"); }},
      {"index", [&] { (void)v3[5]; }},
      {"select", [] { (void)select(true, vec2(1.0), vec3(1.0)); }},
      {"clamp", [] { (void)clamp(0.5, 1.0, 0.0); }},
      {"literal", [] { (void)(toI32(1) + 0.5); }},
      {"constant", [&] { (void)constant("C", v3); }},
      {"statement", [] { (void)Var(1.0); }},
      {"if", [] { If(true, [] {}); }},
      {"switch", [] { Switch(1).Case(1, [] {}); }},
      {"output", [] { (void)output(9); }},
      {"blend output", [] { (void)blendOutput(3); }},
      {"attribute", [] { (void)attribute("b", boolType()); }},
      {"varying", [] { (void)varying("id", u32()); }},
      {"uniform block", [] { (void)uniformBlock("u", {{"a", f32()}, {"a", f32()}}); }},
      {"texture", [] { (void)texture("t", TextureDimension::D3, {.sampleType = TextureSampleType::Depth}); }},
      {"table", [] { (void)table("t", {{"a", f32()}}); }},
      {"storage", [] { (void)storage("s", f32(), StorageAccess::Write); }},
      {"storage texture", [] { (void)storageTexture("s", TextureDimension::Cube, TexelFormat::Rgba8Unorm); }},
      {"workgroup array", [] { (void)workgroupArray("w", f32(), 0); }},
      {"struct", [] { (void)makeStruct("S", {}); }},
      {"array", [] { (void)arrayOf(f32(), 0); }},
      {"function", [] { (void)fn("f", {{"x", voidType()}}, f32(), [](const Args&) { Return(1.0); }); }},
      {"parameter", [] { (void)fn("f", {}, f32(), [](const Args& a) { Return(a["missing"]); }); }},
  };
  const Output out = output(0);
  for (const auto& [what, mistake] : mistakes) {
    CAPTURE(what);
    ShaderBuild build("m", "M");
    CHECK_NOTHROW(mistake());
    const BuildResult r = build.finish([] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); });
    CHECK(r.diagnostics.size() == 1);
  }
}
