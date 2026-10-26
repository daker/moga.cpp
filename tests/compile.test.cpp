#include <doctest/doctest.h>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

using namespace moga;

namespace {
bool has(const std::string& haystack, std::string_view needle) {
  return haystack.find(needle) != std::string::npos;
}
} // namespace

TEST_CASE("minimal program compiles on both backends") {
  const Val pos = attribute("pos", vec3f());
  const Output out = output(0);
  const ProgramAST ast = program(
      "min", "MIN", [&] { Position(vec4(pos, 1.0)); },
      [&] { out.set(vec4(1.0, 0.0, 0.0, 1.0)); });
  const AnalyzedProgram a = analyzeProgram(ast);
  const CompiledGLSL glsl = compileGLSL(a);
  const CompiledWGSL wgsl = compileWGSL(a);

  CHECK(glsl.vertex.starts_with("#version 300 es\n"));
  CHECK(has(glsl.vertex, "layout(location = 0) in vec3 a_pos;"));
  CHECK(has(glsl.vertex, "gl_Position = moga_clipFix(vec4(a_pos, 1.0));"));
  // default options flip Y and remap depth from [0, w] to [-w, w]
  CHECK(has(glsl.vertex, "return vec4(p.x, -p.y, 2.0 * p.z - p.w, p.w);"));
  CHECK(has(glsl.fragment, "o_0 = vec4(1.0, 0.0, 0.0, 1.0);"));

  CHECK(has(wgsl.code, "@location(0) a_pos: vec3f,"));
  CHECK(has(wgsl.code, "vout.position = vec4f(vin.a_pos, 1.0f);"));
  CHECK(has(wgsl.code, "@fragment fn fs_main() -> FSOut {"));
}

TEST_CASE("literals are typed per backend") {
  const Output out = output(0, vec4i());
  const ProgramAST ast = program(
      "lit", "LIT", [] { Position(vec4(0.0)); },
      [&] {
        const Val u = Let(toU32(3), "u");
        out.set(construct(vec4i(), {-2, toI32(u), 0, 1}));
      });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(glsl, " = 3u;"));
  CHECK(has(glsl, "ivec4((-2), int(u_"));
  CHECK(has(wgsl, "vec4i((-2i), i32(u_"));
}

TEST_CASE("backend specific lowering") {
  const Output out = output(0);
  const ProgramAST ast = program(
      "low", "LOW", [] { Position(vec4(0.0)); },
      [&] {
        const Val p = Let(fragCoord().xy(), "p");
        const Val m = Let(mod(p, 4.0), "m");
        const Val c = Let(select(p < vec2(1.0), vec2(0.0), m), "c");
        const Val bits = Let(floatBitsToUint(c.x()), "bits");
        out.set(vec4(c, uintBitsToFloat(bits << toI32(1)), 1.0));
      });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(glsl, "mod(p_"));
  CHECK(has(glsl, "lessThan(p_"));
  CHECK(has(glsl, "mix(m_"));
  CHECK(has(wgsl, "floor(p_"));
  CHECK(has(wgsl, "select(m_"));
  CHECK(has(wgsl, "bitcast<u32>("));
  // WGSL shift amounts must be unsigned
  CHECK(has(wgsl, "<< u32(1i))"));
  CHECK(has(wgsl, "@builtin(position) fragCoord: vec4f,"));
}

TEST_CASE("bindings: groups by frequency, samplers take their own slot") {
  const UniformBlock frameUb = uniformBlock("frame", {{"t", f32()}}, ResourceFrequency::Frame);
  const UniformBlock mat = uniformBlock("mat", {{"c", vec4f()}});
  const Texture tex = texture("albedo", TextureDimension::D2);
  const Texture data =
      texture("data", TextureDimension::D2, {.sampleType = TextureSampleType::Uint});
  const Output out = output(0);
  const ProgramAST ast = program(
      "b", "B", [&] { Position(vec4(frameUb["t"])); },
      [&] {
        const Val d = Let(data.load(ivec2(0, 0)), "d");
        out.set(tex.sample(vec2(0.5)) * mat["c"] + toF32(d.x()));
      });
  const AnalyzedProgram a = analyzeProgram(ast);
  const Reflection& r = a.reflection;
  REQUIRE(r.groups[0].size() == 1);
  REQUIRE(r.groups[2].size() == 3);
  CHECK(r.uniformBlocks[0].name == "frame");
  CHECK(r.uniformBlocks[0].glslBindingPoint == 0);
  CHECK(r.uniformBlocks[1].binding == 0);
  // material group: mat block, then textures by name; albedo has a sampler
  CHECK(r.textures[0].name == "albedo");
  CHECK(r.textures[0].binding == 1);
  CHECK(r.textures[0].samplerBinding == 2);
  CHECK(r.textures[1].binding == 3);
  CHECK(r.textures[1].samplerBinding == -1);
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "@group(2) @binding(2) var s_albedo: sampler;"));
  CHECK(has(wgsl, "@group(2) @binding(3) var t_data: texture_2d<u32>;"));
  CHECK(has(compileGLSL(a).fragment, "uniform highp usampler2D t_data;"));
}

TEST_CASE("uniform structs get explicit WGSL layout") {
  const UniformBlock ub =
      uniformBlock("cam", {{"dir", vec3f()}, {"mvp", mat4f()}}, ResourceFrequency::Frame);
  const ProgramAST ast = program(
      "u", "U", [&] { Position(ub["mvp"] * vec4(ub["dir"], 1.0)); }, [] {});
  const std::string wgsl = compileWGSL(analyzeProgram(ast)).code;
  CHECK(has(wgsl, "@align(16) @size(16) dir: vec3f,"));
  CHECK(has(wgsl, "@align(16) @size(64) mvp: mat4x4f,"));
}

TEST_CASE("tables lower to texel fetches and storage buffers") {
  const Table inst =
      table("inst", {{"tint", vec4f()}, {"model", mat4f()}}, ResourceFrequency::Object);
  const ProgramAST ast = program(
      "tb", "TB",
      [&] { Position(inst.at(instanceIndex())["model"] * inst.at(instanceIndex())["tint"]); },
      [] {});
  const AnalyzedProgram a = analyzeProgram(ast);
  const CompiledGLSL glsl = compileGLSL(a);
  CHECK(has(glsl.vertex, "int t = int(record) * 5 + texel;"));
  CHECK(has(glsl.vertex, "tb_inst_fetch(uint(gl_InstanceID + moga_BaseInstance), 4))"));
  CHECK(glsl.baseInstanceUniform == "moga_BaseInstance");
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "var<storage, read> tb_inst: array<S_instRecord>;"));
  CHECK(has(wgsl, "tb_inst[vin.instanceIndex].model"));
}

TEST_CASE("functions are emitted callees first") {
  const Function sq =
      fn("sq", {{"x", f32()}}, f32(), [](const Args& a) { Return(a["x"] * a["x"]); });
  const Function quad =
      fn("quad", {{"x", f32()}}, f32(), [&](const Args& a) { Return(sq(sq(a["x"]))); });
  const Output out = output(0);
  const ProgramAST ast = program(
      "f", "F", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(quad(2.0))); });
  const std::string wgsl = compileWGSL(analyzeProgram(ast)).code;
  const auto sqAt = wgsl.find("fn f_sq(");
  const auto quadAt = wgsl.find("fn f_quad(");
  REQUIRE(sqAt != std::string::npos);
  REQUIRE(quadAt != std::string::npos);
  CHECK(sqAt < quadAt);
  CHECK(has(wgsl, "f_sq(f_sq(x_"));
}

TEST_CASE("analysis rejects invalid stage interfaces") {
  const Varying v = varying("c", vec3f());
  const Output out = output(0);
  const ProgramAST unwritten = program(
      "x", "X", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(v.value(), 1.0)); });
  CHECK_THROWS_AS(analyzeProgram(unwritten), ShaderError);

  const Val a = attribute("a", f32());
  const Function leaky = fn("leaky", {}, f32(), [&](const Args&) { Return(a); });
  const ProgramAST leaking = program("y", "Y", [&] { Position(vec4(leaky())); }, [] {});
  CHECK_THROWS_AS(analyzeProgram(leaking), ShaderError);
}

TEST_CASE("GLSL only builtins need their extension") {
  const ProgramAST ast =
      program("d", "D", [] { Position(vec4(toF32(drawIndex()))); }, [] {});
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK_THROWS_AS(compileGLSL(a), ShaderError);
  CHECK(has(compileGLSL(a, {.multiDraw = true}).vertex, "#extension GL_ANGLE_multi_draw : require"));
  CHECK_THROWS_AS(compileWGSL(a), ShaderError);
}

TEST_CASE("debug mode maps generated lines to DSL statements") {
  const Output out = output(0);
  const std::uint32_t line = __LINE__ + 2;
  const BuildResult built = buildShader(
      "dbg", "DBG", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0));
      }, {.wgsl = false, .sourceLocations = true});
  REQUIRE(built.glsl);
  const CompiledGLSL& glsl = *built.glsl;
  REQUIRE(glsl.fragmentLineMap.size() == 1);
  CHECK(glsl.fragmentLineMap[0].debug.line == line);
  // the mapped line is the output write
  std::string_view text = glsl.fragment;
  for (int i = 1; i < glsl.fragmentLineMap[0].line; ++i) text.remove_prefix(text.find('\n') + 1);
  CHECK(text.starts_with("  o_0 = vec4(1.0);"));
}

TEST_CASE("texture kinds used by post processing passes") {
  const Texture depth = texture("depth", TextureDimension::D2,
                                {.sampleType = TextureSampleType::Depth, .sampled = false});
  const Texture color = texture("color", TextureDimension::D2,
                                {.sampleType = TextureSampleType::UnfilterableFloat});
  const Texture env = texture("env", TextureDimension::Cube);
  const Texture lut = texture("lut", TextureDimension::D3);
  const Output out = output(0);
  const ProgramAST ast = program(
      "post", "POST", [] { Position(vec4(0.0)); },
      [&] {
        const Val p = Let(cast(fragCoord().xy(), ScalarKind::I32), "p");
        const Val d = Let(depth.load(p), "d");
        const Val size = Let(color.size(), "size");
        const Val c = Let(color.load(p).rgb() + env.sample(vec3(0.0, 1.0, 0.0)).rgb() +
                              lut.sample(vec3(0.5)).rgb() + toF32(size.x()),
                          "c");
        FragDepth(d);
        out.set(vec4(c, 1.0));
      });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "var t_depth: texture_depth_2d;"));
  CHECK_FALSE(has(wgsl, "s_depth"));
  CHECK(has(wgsl, "var t_env: texture_cube<f32>;"));
  CHECK(has(wgsl, "var t_lut: texture_3d<f32>;"));
  // unfilterable textures have no sampler by default
  CHECK_FALSE(has(wgsl, "s_color"));
  CHECK(has(wgsl, "@builtin(frag_depth) fragDepth: f32,"));
  CHECK(has(wgsl, "vec2i(textureDimensions(t_color, 0i))"));
  // a depth texel load returns one float in GLSL
  CHECK(has(glsl, "texelFetch(t_depth, p_"));
  CHECK(has(glsl, ", 0).x;"));
  CHECK(has(glsl, "uniform highp samplerCube t_env;"));
  CHECK(has(glsl, "gl_FragDepth = d_"));
}

TEST_CASE("retained resources keep the frame group layout") {
  const UniformBlock camera =
      uniformBlock("camera", {{"viewProj", mat4f()}}, ResourceFrequency::Frame);
  const UniformBlock lights =
      uniformBlock("lights", {{"intensity", f32()}}, ResourceFrequency::Frame);
  const ProgramAST ast = program("r", "R", [&] { Position(vec4(lights["intensity"])); }, [] {});

  const AnalyzedProgram alone = analyzeProgram(ast);
  REQUIRE(alone.reflection.uniformBlocks.size() == 1);
  CHECK(alone.reflection.uniformBlocks[0].binding == 0);

  // camera sorts before lights, so retaining it moves lights to binding 1
  const AnalyzedProgram shared = analyzeProgram(ast, {.retain = {camera.resource()}});
  REQUIRE(shared.reflection.uniformBlocks.size() == 2);
  CHECK(shared.reflection.uniformBlocks[0].name == "camera");
  CHECK(shared.reflection.uniformBlocks[1].binding == 1);
  CHECK(has(compileWGSL(shared).code, "@group(0) @binding(0) var<uniform> ub_camera"));
}

TEST_CASE("clip control keeps depth in zero to one") {
  const ProgramAST ast = program("c", "C", [] { Position(vec4(0.0)); }, [] {});
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(has(compileGLSL(a, {.clipControl = true}).vertex, "return vec4(p.x, -p.y, p.z, p.w);"));
  CHECK(has(compileGLSL(a, {.flipY = false, .clipControl = true}).vertex,
            "return vec4(p.x, p.y, p.z, p.w);"));
}

TEST_CASE("named constants are declared once, dependencies first") {
  const Val tau = constant("TAU", consts::pi() * 2.0);
  const Output out = output(0);
  const ProgramAST ast = program(
      "k", "K", [] { Position(vec4(0.0)); },
      [&] { out.set(vec4(tau + consts::pi() + consts::infinity())); });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  const std::string wgsl = compileWGSL(a).code;
  // PI is used by TAU, so it is declared before TAU and only one time
  const auto piAt = glsl.find("const float PI = 3.141592653589793;");
  const auto tauAt = glsl.find("const float TAU = (PI * 2.0);");
  REQUIRE(piAt != std::string::npos);
  REQUIRE(tauAt != std::string::npos);
  CHECK(piAt < tauAt);
  CHECK(glsl.find("const float PI", piAt + 1) == std::string::npos);
  CHECK(has(wgsl, "const TAU: f32 = (PI * 2.0f);"));
  // 0x7F800000 is the bit pattern of +inf
  CHECK(has(glsl, "const float INFINITY = uintBitsToFloat(2139095040u);"));
  CHECK(has(wgsl, "var bits: u32 = 2139095040u;"));
  CHECK(has(wgsl, "+ INFINITY())"));
  CHECK_FALSE(has(wgsl, "HALF_PI"));
}

TEST_CASE("constants must be known at compile time") {
  const UniformBlock ub = uniformBlock("u", {{"x", f32()}});
  CHECK_THROWS_AS(constant("BAD", ub["x"] * 2.0), ShaderError);
  CHECK_THROWS_AS(constant("BAD", attribute("a", f32())), ShaderError);
  CHECK_NOTHROW(constant("OK", vec3(consts::halfPi(), 1.0, 2.0).x()));
}

TEST_CASE("local names never contain two underscores in a row") {
  const Output out = output(0);
  const ProgramAST ast = program(
      "n", "N", [] { Position(vec4(0.0)); },
      [&] { out.set(vec4(Let(1.0, "packed") + Let(2.0, "a__b") + Let(3.0, "x_"))); });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK_FALSE(has(compileGLSL(a).fragment, "__"));
  CHECK_FALSE(has(compileWGSL(a).code, "__"));
}
