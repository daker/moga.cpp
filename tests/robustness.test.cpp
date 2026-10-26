#include <doctest/doctest.h>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cstring>
#include <map>
#include <optional>

using namespace moga;

namespace {

bool has(const std::string& haystack, std::string_view needle) {
  return haystack.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("uniform writer rejects writes that do not match the field") {
  const auto l = structLayout(makeStruct(
      "W", {{"color", vec4f()}, {"dir", vec3f()}, {"count", u32()}, {"lights", arrayOf(vec4f(), 2)}}));
  UniformWriter w(l);
  const std::array<float, 4> four{1, 2, 3, 4};
  const std::array<float, 16> m{};

  CHECK_THROWS_AS(w.setArrayVec4("lights", 2, 0, 0, 0, 0), ShaderError);
  CHECK_THROWS_AS(w.setArrayVec4("color", 0, 0, 0, 0, 0), ShaderError);
  CHECK_THROWS_AS(w.setVec("dir", four), ShaderError);
  CHECK_THROWS_AS(w.setMat4("color", m), ShaderError);
  CHECK_THROWS_AS(w.setFloat("count", 1.0f), ShaderError);
  CHECK_THROWS_AS(w.setUint("dir", 1u), ShaderError);

  // valid writes land at the std140 offsets: dir at 16, count at 28, lights at 32
  w.setVec("dir", std::span<const float>(four.data(), 3));
  w.setUint("count", 7u);
  w.setArrayVec4("lights", 1, 5, 6, 7, 8);
  float dir[3];
  std::uint32_t count = 0;
  float light[4];
  std::memcpy(dir, w.bytes().data() + 16, sizeof dir);
  std::memcpy(&count, w.bytes().data() + 28, sizeof count);
  std::memcpy(light, w.bytes().data() + 48, sizeof light);
  CHECK(dir[2] == 3.0f);
  CHECK(count == 7u);
  CHECK(light[3] == 8.0f);
}

TEST_CASE("an operation on two literals makes f32 literals") {
  const Val sum = binary(BinaryOp::Add, 1, 2);
  CHECK(sum.type() == f32());
  CHECK(binary(BinaryOp::Lt, 1, 2).type() == boolType());
}

TEST_CASE("texture layers are checked against the dimension") {
  const Texture flat = texture("flat", TextureDimension::D2);
  const Texture layers = texture("layers", TextureDimension::D2Array);
  const Output out = output(0);
  program("l", "L", [] { Position(vec4(0.0)); }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    CHECK_THROWS_AS(layers.sample(uv), ShaderError);
    CHECK_THROWS_AS(flat.sample(uv, 0, 1), ShaderError);
    CHECK_THROWS_AS(layers.load(ivec2(0, 0)), ShaderError);
    CHECK_THROWS_AS(flat.load(ivec2(0, 0), 0, 1), ShaderError);
    CHECK_THROWS_AS(layers.sampleGrad(uv, uv, uv), ShaderError);
    CHECK_THROWS_AS(layers.sample(uv, 0, 1.5), ShaderError);
    out.set(layers.sample(uv, 0, 2));
  });
}

TEST_CASE("depth textures sample one f32 with an integer level") {
  const Texture depth = texture("depth", TextureDimension::D2,
                                {.sampleType = TextureSampleType::Depth, .sampled = true});
  const Output out = output(0);
  const ProgramAST ast = program("d", "D", [] { Position(vec4(0.0)); }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    const Val d = Let(depth.sample(uv), "d");
    CHECK(d.type() == f32());
    CHECK_THROWS_AS(depth.sample(uv, 0.5), ShaderError);
    CHECK_THROWS_AS(depth.sampleGrad(uv, uv, uv), ShaderError);
    out.set(vec4(d));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  // WGSL: textureSampleLevel on texture_depth_2d takes an integer level and returns f32
  CHECK(has(compileWGSL(a).code, "textureSampleLevel(t_depth, s_depth, uv_"));
  CHECK(has(compileWGSL(a).code, ", 0i);"));
  // GLSL: a float level and the first component
  CHECK(has(compileGLSL(a).fragment, "textureLod(t_depth, uv_"));
  CHECK(has(compileGLSL(a).fragment, ", float(0)).x;"));
}

TEST_CASE("structs with the same name and different fields are different types") {
  const TypePtr a = makeStruct("Light", {{"color", vec4f()}});
  const TypePtr b = makeStruct("Light", {{"color", vec3f()}});
  const TypePtr c = makeStruct("Light", {{"color", vec4f()}});
  CHECK_FALSE(typesEqual(a, b));
  CHECK(typesEqual(a, c));

  const Function useA = fn("useA", {{"l", a}}, vec4f(), [](const Args& args) {
    Return(args["l"].member("color"));
  });
  const Output out = output(0);
  program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    CHECK_THROWS_AS(useA(construct(b, {vec3(1.0)})), ShaderError);
    out.set(useA(construct(c, {vec4(1.0)})));
  });

  // two definitions of one name in one program cannot both be declared
  const UniformBlock ub = uniformBlock("u", {{"x", a}, {"y", b}});
  const ProgramAST clash = program("c", "C", [&] { Position(ub["x"].member("color")); }, [] {});
  CHECK_THROWS_AS(analyzeProgram(clash), ShaderError);

  // equal definitions are declared once
  const UniformBlock same = uniformBlock("v", {{"x", a}, {"y", c}});
  const ProgramAST ok = program("o", "O", [&] { Position(same["y"].member("color")); }, [] {});
  const std::string wgsl = compileWGSL(analyzeProgram(ok)).code;
  const auto first = wgsl.find("struct S_Light {");
  REQUIRE(first != std::string::npos);
  CHECK(wgsl.find("struct S_Light {", first + 1) == std::string::npos);
  CHECK(has(wgsl, "@align(16) @size(16) color: vec4f,"));
}

TEST_CASE("integer textures are loaded, never sampled") {
  CHECK_THROWS_AS(texture("ids", TextureDimension::D2,
                          {.sampleType = TextureSampleType::Uint, .sampled = true}),
                  ShaderError);
  CHECK_THROWS_AS(texture("offsets", TextureDimension::D3,
                          {.sampleType = TextureSampleType::Sint, .sampled = true}),
                  ShaderError);
  const Texture ids = texture("ids", TextureDimension::D2, {.sampleType = TextureSampleType::Uint});
  const Output out = output(0);
  const ProgramAST ast = program("i", "I", [] { Position(vec4(0.0)); }, [&] {
    CHECK_THROWS_AS(ids.sample(vec2(0.5)), ShaderError);
    CHECK_THROWS_AS(ids.sampleOffset(vec2(0.5), ivec2(1, 0)), ShaderError);
    out.set(vec4(toF32(ids.load(ivec2(0, 0)).x())));
  });
  const std::string wgsl = compileWGSL(analyzeProgram(ast)).code;
  CHECK(has(wgsl, "var t_ids: texture_2d<u32>;"));
  CHECK(has(wgsl, "textureLoad(t_ids, vec2i(0i, 0i), 0i)"));
  CHECK_FALSE(has(wgsl, "s_ids"));
}

TEST_CASE("derivatives are fragment only, also through functions") {
  CHECK_THROWS_AS(program("v", "V", [] { Position(vec4(dpdx(1.0))); }, [] {}), ShaderError);
  CHECK_THROWS_AS(computeProgram("c", "C", {1, 1, 1}, [] { (void)fwidth(1.0); }), ShaderError);

  const Function edge =
      fn("edge", {{"x", f32()}}, f32(), [](const Args& a) { Return(fwidth(a["x"])); });
  const Output out = output(0);
  const ProgramAST inFragment =
      program("f", "F", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(edge(1.0))); });
  CHECK(has(compileWGSL(analyzeProgram(inFragment)).code, "fwidth(x_"));
  const ProgramAST inVertex = program("v", "V", [&] { Position(vec4(edge(1.0))); }, [] {});
  CHECK_THROWS_AS(analyzeProgram(inVertex), ShaderError);
}

TEST_CASE("return values must match the function") {
  CHECK_THROWS_AS(fn("f", {}, f32(), [](const Args&) { Return(vec4(1.0)); }), ShaderError);
  CHECK_THROWS_AS(fn("f", {}, f32(), [](const Args&) { Return(); }), ShaderError);
  CHECK_THROWS_AS(fn("f", {}, voidType(), [](const Args&) { Return(1.0); }), ShaderError);
  CHECK_THROWS_AS(program("p", "P", [] { Return(1.0); }, [] {}), ShaderError);
  // a literal takes the declared return type
  const Function count = fn("count", {}, u32(), [](const Args&) { Return(3); });
  const Output out = output(0);
  const ProgramAST ast =
      program("c", "C", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(toF32(count()))); });
  CHECK(has(compileWGSL(analyzeProgram(ast)).code, "return 3u;"));
}

TEST_CASE("select needs a bool condition of the right shape") {
  const Output out = output(0);
  program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec3(1.0), "v");
    CHECK_THROWS_AS(select(1.0, 2.0, 3.0), ShaderError);
    CHECK_THROWS_AS(select(vec2(1.0) < vec2(2.0), v, v), ShaderError);
    CHECK_THROWS_AS(select(v < v, 1.0, 2.0), ShaderError);
    CHECK_NOTHROW(select(v < v, v, v));
    out.set(vec4(select(fragCoord().x() > 1.0, v, v), 1.0));
  });
}

TEST_CASE("clip distance indices are in 0..7") {
  CHECK_THROWS_AS(program("c", "C", [] { ClipDistance(-1, 1.0); }, [] {}), ShaderError);
  CHECK_THROWS_AS(program("c", "C", [] { ClipDistance(8, 1.0); }, [] {}), ShaderError);
  const ProgramAST ast = program("c", "C", [] {
    ClipDistance(7, 1.0);
    Position(vec4(0.0));
  }, [] {});
  const Reflection r = analyzeProgram(ast).reflection;
  CHECK(r.clipDistances == 8);
  // WGSL enable clip_distances needs the optional WebGPU device feature
  CHECK(r.deviceFeatures == std::vector{DeviceFeature::ClipDistances});
  CHECK(deviceFeatureName(DeviceFeature::ClipDistances) == "clip-distances");
  const ProgramAST none = program("n", "N", [] { Position(vec4(0.0)); }, [] {});
  CHECK(analyzeProgram(none).reflection.deviceFeatures.empty());
}

TEST_CASE("GLSL selects integer and bool vectors per component") {
  const Output out = output(0, vec4i());
  const ProgramAST ast = program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val c = Let(bvec2(true, false), "c");
    const Val i = Let(select(c, ivec2(1, 2), ivec2(3, 4)), "i");
    const Val b = Let(select(c, bvec2(true, true), bvec2(false, false)), "b");
    const Val f = Let(select(c, vec2(1.0), vec2(0.0)), "f");
    out.set(ivec4(i, toIVec2(f)) * toI32(b.x()));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  // GLSL ES 3.00 has mix(x, y, bvec) for float types only
  CHECK(has(glsl, "ivec2(((c_"));
  CHECK(has(glsl, "bvec2(((c_"));
  CHECK(has(glsl, "mix(vec2(0.0), vec2(1.0), c_"));
  CHECK_FALSE(has(glsl, "mix(ivec2"));
  CHECK(has(compileWGSL(a).code, "select(vec2i(3i, 4i), vec2i(1i, 2i), c_"));
}

TEST_CASE("two resources cannot get the same shader name") {
  const Texture a = texture("x", TextureDimension::D2);
  const Texture b = texture("x", TextureDimension::D2);
  const Texture dashed = texture("a-b", TextureDimension::D2);
  const Texture underscored = texture("a_b", TextureDimension::D2);
  const Output out = output(0);
  const auto both = [&](const Texture& p, const Texture& q) {
    return program("r", "R", [] { Position(vec4(0.0)); },
                   [&] { out.set(p.sample(vec2(0.5)) + q.sample(vec2(0.5))); });
  };
  CHECK_THROWS_AS(analyzeProgram(both(a, b)), ShaderError);
  // the names differ, but both sanitize to t_a_b
  CHECK_THROWS_AS(analyzeProgram(both(dashed, underscored)), ShaderError);
  // one resource used twice is fine
  CHECK_NOTHROW(analyzeProgram(both(a, a)));
  // different kinds have different prefixes
  const UniformBlock ub = uniformBlock("x", {{"v", vec4f()}});
  const ProgramAST mixed = program("m", "M", [] { Position(vec4(0.0)); },
                                   [&] { out.set(a.sample(vec2(0.5)) + ub["v"]); });
  CHECK_NOTHROW(analyzeProgram(mixed));
}

TEST_CASE("math builtins take only the types the backends define") {
  const Output out = output(0);
  program("m", "M", [] { Position(vec4(0.0)); }, [&] {
    const Val m = Let(mat3(1.0), "m");
    CHECK_THROWS_AS(floor(m), ShaderError);
    CHECK_THROWS_AS(sin(m), ShaderError);
    CHECK_THROWS_AS(normalize(1.0), ShaderError);
    CHECK_THROWS_AS(abs(toBool(true)), ShaderError);
    CHECK_THROWS_AS(min(bvec2(true, false), bvec2(true, true)), ShaderError);
    CHECK_THROWS_AS(mod(toI32(5), toI32(3)), ShaderError);
    CHECK_THROWS_AS(pow(toI32(2), toI32(3)), ShaderError);
    CHECK_THROWS_AS(mix(toI32(1), toI32(2), 0.5), ShaderError);
    CHECK_THROWS_AS(dot(bvec2(true, true), bvec2(true, true)), ShaderError);
    CHECK_THROWS_AS(reflect(vec3(1.0), vec2(1.0)), ShaderError);
    CHECK_THROWS_AS(transpose(vec3(1.0)), ShaderError);
    CHECK_THROWS_AS(determinant(vec3(1.0)), ShaderError);
    CHECK_THROWS_AS(any(vec2(1.0)), ShaderError);
    // valid integer and scalar uses still work
    CHECK(abs(toI32(-2)).type() == i32());
    CHECK(min(toU32(1), toU32(2)).type() == u32());
    CHECK(clamp(ivec2(5, 6), 0, 3).type() == vec2i());
    CHECK(length(1.0).type() == f32());
    CHECK(any(toBool(true)).type() == boolType());
    out.set(vec4(1.0));
  });
}

TEST_CASE("Break and Continue need an enclosing loop or switch") {
  const Output out = output(0);
  CHECK_THROWS_AS(program("b", "B", [] { Break(); }, [] {}), ShaderError);
  CHECK_THROWS_AS(program("c", "C", [] { Continue(); }, [] {}), ShaderError);
  // Continue needs a loop; a switch alone is not enough
  CHECK_THROWS_AS(program("s", "S", [] { Switch(toI32(1)).Case(1, [] { Continue(); }); }, [] {}),
                  ShaderError);
  // a function body does not see the loop around its definition
  CHECK_THROWS_AS(program("f", "F", [] {
    For(0, 2, [](const Val&) { fn("g", {}, voidType(), [](const Args&) { Break(); }); });
  }, [] {}),
                  ShaderError);
  const ProgramAST ok = program("o", "O", [] { Position(vec4(0.0)); }, [&] {
    const Val s = Var(0.0, "s");
    For(0, 4, [&](const Val& i) {
      Switch(i).Case(1, [] { Continue(); }).Case(2, [] { Break(); });
      If(i > 2, [] { Break(); });
      s.addAssign(toF32(i));
    });
    out.set(vec4(s));
  });
  const AnalyzedProgram a = analyzeProgram(ok);
  CHECK(has(compileWGSL(a).code, "continue;"));
  CHECK(has(compileGLSL(a).fragment, "continue;"));
}

TEST_CASE("GLSL writes out the overloads that only WGSL has") {
  const Output out = output(0);
  const ProgramAST ast = program("o", "O", [] { Position(vec4(0.0)); }, [&] {
    const Val u = Let(toU32(fragCoord().x()), "u");
    const Val a = Let(ivec2(1, 2), "a");
    const Val flag = Let(u > 2u, "flag");
    const Val r = Let(toF32(abs(u)) + toF32(dot(a, a)) + toF32(toI32(any(flag))) +
                          toF32(toI32(all(flag))),
                      "r");
    out.set(vec4(r));
  });
  const AnalyzedProgram an = analyzeProgram(ast);
  const std::string glsl = compileGLSL(an).fragment;
  // GLSL ES 3.00 has no abs(uint), integer dot, or any/all on a bool scalar
  CHECK_FALSE(has(glsl, "abs(u_"));
  CHECK_FALSE(has(glsl, "dot(a_"));
  CHECK_FALSE(has(glsl, "any(flag_"));
  CHECK_FALSE(has(glsl, "all(flag_"));
  CHECK(has(glsl, "((l_a_"));
  CHECK(has(glsl, ").x * (l_a_"));
  // WGSL keeps the builtins
  const std::string wgsl = compileWGSL(an).code;
  CHECK(has(wgsl, "abs(u_"));
  CHECK(has(wgsl, "dot(l_a_"));
  CHECK(has(wgsl, "any(flag_"));
}

TEST_CASE("texture levels and layers must be integers") {
  const Texture flat = texture("flat", TextureDimension::D2);
  const StorageTexture layers = storageTexture("layers", TextureDimension::D2Array,
                                               TexelFormat::R32Float, StorageAccess::ReadWrite);
  const Output out = output(0);
  program("t", "T", [] { Position(vec4(0.0)); }, [&] {
    const Val f = Let(fragCoord().x(), "f");
    CHECK_THROWS_AS(flat.load(ivec2(0, 0), f), ShaderError);
    CHECK_THROWS_AS(flat.size(f), ShaderError);
    CHECK_NOTHROW(flat.load(ivec2(0, 0), toI32(f)));
    out.set(vec4(1.0));
  });
  computeProgram("c", "C", {1, 1, 1}, [&] {
    const Val f = Let(1.5, "f");
    CHECK_THROWS_AS(layers.load(uvec2(0, 0), f), ShaderError);
    CHECK_THROWS_AS(layers.store(uvec2(0, 0), vec4(0.0), f), ShaderError);
    CHECK_NOTHROW(layers.store(uvec2(0, 0), vec4(0.0), 1u));
  });
}

TEST_CASE("switch case values are unique") {
  program("s", "S", [] { Position(vec4(0.0)); }, [] {
    const Val i = Let(toI32(fragCoord().x()), "i");
    CHECK_THROWS_AS(Switch(i).Case(1, [] {}).Case(1, [] {}), ShaderError);
    CHECK_THROWS_AS(Switch(i).Case({2, 2}, [] {}), ShaderError);
    CHECK_THROWS_AS(Switch(i).Case({3, 4}, [] {}).Case({5, 3}, [] {}), ShaderError);
    CHECK_NOTHROW(Switch(i).Case({1, 2}, [] {}).Case(3, [] {}));
  });
}

TEST_CASE("a cached layout does not keep its struct alive, but a held layout does") {
  std::weak_ptr<const TypeInfo> weak;
  {
    const TypePtr s = makeStruct("Temp", {{"x", vec4f()}});
    (void)structLayout(s);
    weak = s;
  }
  CHECK(weak.expired());
  // the returned layout shares ownership of the struct, so this does not dangle
  const auto held = structLayout(makeStruct("Held", {{"x", f32()}, {"y", vec4f()}}));
  CHECK(held->structType->name == "Held");
  CHECK(held->find("y")->offset == 16);
}

TEST_CASE("module scope names never collide") {
  const Output out = output(0);
  const auto render = [&](const std::function<void()>& body) {
    return program("n", "N", [] { Position(vec4(0.0)); }, [&] {
      body();
      out.set(vec4(1.0));
    });
  };
  const Function a = fn("foo", {}, f32(), [](const Args&) { Return(1.0); });
  const Function b = fn("foo", {}, f32(), [](const Args&) { Return(2.0); });
  CHECK_THROWS_AS(analyzeProgram(render([&] { (void)Let(a() + b(), "s"); })), ShaderError);
  // one function called twice is fine
  CHECK_NOTHROW(analyzeProgram(render([&] { (void)Let(a() + a(), "s"); })));

  const Val k1 = constant("K-1", 1.0);
  const Val k2 = constant("K_1", 2.0);
  CHECK_THROWS_AS(analyzeProgram(render([&] { (void)Let(k1 + k2, "s"); })), ShaderError);
  // a name the backends generate
  const Val clash = constant("VSOut", 1.0);
  CHECK_THROWS_AS(analyzeProgram(render([&] { (void)Let(clash, "s"); })), ShaderError);
}

TEST_CASE("resource, attribute and varying names join the module name check") {
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Val clash = constant("t_albedo", 1.0);
  const Output out = output(0);
  const ProgramAST resource = program("r", "R", [] { Position(vec4(0.0)); },
                                      [&] { out.set(albedo.sample(vec2(0.5)) * clash); });
  CHECK_THROWS_AS(analyzeProgram(resource), ShaderError);

  const Val a = attribute("pos-x", f32());
  const Val b = attribute("pos_x", f32());
  const ProgramAST attributes =
      program("a", "A", [&] { Position(vec4(a + b)); }, [] {});
  CHECK_THROWS_AS(analyzeProgram(attributes), ShaderError);

  const Varying v1 = varying("uv.0", vec2f());
  const Varying v2 = varying("uv_0", vec2f());
  const ProgramAST varyings = program(
      "v", "V",
      [&] {
        v1.set(vec2(0.0));
        v2.set(vec2(1.0));
        Position(vec4(0.0));
      },
      [&] { out.set(vec4(v1.value() + v2.value(), 0.0, 1.0)); });
  CHECK_THROWS_AS(analyzeProgram(varyings), ShaderError);
}

TEST_CASE("GLSL resource names are sanitized like the code that reads them") {
  const Texture tex = texture("base-color", TextureDimension::D2);
  const UniformBlock ub = uniformBlock("my-block", {{"tint", vec4f()}});
  const Output out = output(0);
  const ProgramAST ast = program("g", "G", [] { Position(vec4(0.0)); },
                                 [&] { out.set(tex.sample(vec2(0.5)) * ub["tint"]); });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(a.reflection.textures[0].glslName == "t_base_color");
  CHECK(a.reflection.uniformBlocks[0].glslBlockName == "UB_my_block");
  const std::string glsl = compileGLSL(a).fragment;
  CHECK(has(glsl, "uniform highp sampler2D t_base_color;"));
  CHECK(has(glsl, "textureLod(t_base_color, "));
  CHECK_FALSE(has(glsl, "base-color"));
}

TEST_CASE("math operands must have matching shapes") {
  const Output out = output(0);
  program("m", "M", [] { Position(vec4(0.0)); }, [&] {
    const Val a2 = Let(vec2(1.0), "a2");
    const Val a3 = Let(vec3(1.0), "a3");
    CHECK_THROWS_AS(distance(a2, a3), ShaderError);
    CHECK_THROWS_AS(clamp(a3, a2, a3), ShaderError);
    CHECK_THROWS_AS(mix(a3, a3, a2), ShaderError);
    CHECK_THROWS_AS(mix(a3, a2, 0.5), ShaderError);
    CHECK_THROWS_AS(smoothstep(a2, a3, a3), ShaderError);
    CHECK_THROWS_AS(step(a2, a3), ShaderError);
    CHECK_THROWS_AS(clamp(a3, toI32(0), 1.0), ShaderError);
    // scalars still widen
    CHECK(clamp(a3, 0.0, 1.0).type() == vec3f());
    CHECK(mix(a3, a3, 0.5).type() == vec3f());
    CHECK(smoothstep(0.0, 1.0, a2).type() == vec2f());
    out.set(vec4(1.0));
  });
}

TEST_CASE("sample levels and depth references must be f32") {
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Texture shadow = texture("shadow", TextureDimension::D2,
                                 {.sampleType = TextureSampleType::Depth, .comparison = true});
  const Output out = output(0);
  program("l", "L", [] { Position(vec4(0.0)); }, [&] {
    const Val i = Let(toI32(fragCoord().x()), "i");
    const Val uv = Let(vec2(0.5), "uv");
    CHECK_THROWS_AS(albedo.sample(uv, i), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ivec2(1, 0), i), ShaderError);
    CHECK_THROWS_AS(shadow.compare(uv, i), ShaderError);
    CHECK_NOTHROW(albedo.sample(uv, 1));
    out.set(albedo.sample(uv, toF32(i)) * shadow.compare(uv, 0.5));
  });
}

TEST_CASE("WGSL writes a swizzle one component at a time") {
  const Output out = output(0);
  const ProgramAST ast = program("w", "W", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Var(vec4(0.0), "v");
    v.swizzle("zx").set(vec2(1.0, 2.0));
    v.y().set(3.0);
    CHECK_THROWS_AS(v.swizzle("xx").set(vec2(1.0)), ShaderError);
    const Val s = Var(1.0, "s");
    CHECK_THROWS_AS(s.swizzle("xx").set(vec2(1.0)), ShaderError);
    out.set(v);
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "let moga_swizzle = vec2f(1.0f, 2.0f);"));
  CHECK(has(wgsl, ".z = moga_swizzle.x;"));
  CHECK(has(wgsl, ".x = moga_swizzle.y;"));
  CHECK(has(wgsl, ".y = 3.0f;"));
  // GLSL writes swizzles natively
  CHECK(has(compileGLSL(a).fragment, ".zx = vec2(1.0, 2.0);"));
}

TEST_CASE("bitwise and shift operands have matching shapes") {
  const Output out = output(0);
  program("b", "B", [] { Position(vec4(0.0)); }, [&] {
    const Val u2 = Let(uvec2(1, 2), "u2");
    const Val i3 = Let(ivec3(1, 2, 3), "i3");
    CHECK_THROWS_AS(u2 & i3, ShaderError);
    CHECK_THROWS_AS(u2 | toI32(1), ShaderError);
    CHECK_THROWS_AS(u2 << ivec3(1, 1, 1), ShaderError);
    // a scalar of the same kind widens; a shift amount may have any integer kind
    CHECK((u2 & 1u).type() == vec2u());
    CHECK((1u ^ u2).type() == vec2u());
    CHECK((u2 << toI32(1)).type() == vec2u());
    out.set(vec4(1.0));
  });
}

TEST_CASE("ordered comparisons need numbers, equality takes bools") {
  const Output out = output(0);
  program("c", "C", [] { Position(vec4(0.0)); }, [&] {
    const Val t = Let(toBool(true), "t");
    CHECK_THROWS_AS(t < t, ShaderError);
    CHECK_THROWS_AS(bvec2(true, false) >= bvec2(true, true), ShaderError);
    CHECK(t.eq(t).type() == boolType());
    CHECK(bvec2(true, false).ne(bvec2(true, true)).type() == vec2b());
    out.set(vec4(1.0));
  });
}

TEST_CASE("refract takes a scalar ratio and arrays a positive length") {
  const Output out = output(0);
  program("r", "R", [] { Position(vec4(0.0)); }, [&] {
    const Val n = Let(vec3(0.0, 1.0, 0.0), "n");
    CHECK_THROWS_AS(refract(n, n, vec3(0.5)), ShaderError);
    CHECK(refract(n, n, 0.5).type() == vec3f());
    out.set(vec4(1.0));
  });
  CHECK_THROWS_AS(arrayOf(f32(), 0), ShaderError);
  CHECK_THROWS_AS(arrayOf(f32(), -1), ShaderError);
}

TEST_CASE("integer literals must fit their type") {
  const Output out = output(0);
  program("l", "L", [] { Position(vec4(0.0)); }, [&] {
    CHECK_THROWS_AS(toI32(2147483648.0), ShaderError);
    CHECK_THROWS_AS(toI32(-2147483649.0), ShaderError);
    CHECK_THROWS_AS(toU32(-1), ShaderError);
    CHECK_THROWS_AS(toU32(4294967296.0), ShaderError);
    CHECK_THROWS_AS(toU32(1e300), ShaderError);
    CHECK_THROWS_AS(toF32(1e39), ShaderError);
    CHECK_NOTHROW(toI32(-2147483648.0));
    CHECK_NOTHROW(toU32(4294967295.0));
    out.set(vec4(toF32(toU32(4294967295.0))));
  });
}

TEST_CASE("a swizzle of a swizzle is one swizzle of the base") {
  const Output out = output(0);
  const StorageBuffer data = storage("data", vec4f());
  const ProgramAST ast = program("n", "N", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Var(vec4(1.0, 2.0, 3.0, 4.0), "v");
    const Val yx = v.xy().swizzle("yx");
    CHECK(std::get<ex::Swizzle>(yx.node()->node).comps == "yx");
    CHECK(std::holds_alternative<ex::Local>(std::get<ex::Swizzle>(yx.node()->node).base->node));
    yx.set(vec2(5.0, 6.0));
    data[toU32(fragCoord().x())].zw().swizzle("yx").set(vec2(7.0, 8.0));
    out.set(v);
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string wgsl = compileWGSL(a).code;
  // one pointer to the base, then one component at a time
  CHECK(has(wgsl, "let moga_target = &l_v_"));
  CHECK(has(wgsl, "(*moga_target).y = moga_swizzle.x;"));
  CHECK(has(wgsl, "let moga_target = &sb_data[u32(fin.fragCoord.x)];"));
  CHECK(has(wgsl, "(*moga_target).w = moga_swizzle.x;"));
  CHECK_FALSE(has(wgsl, ".xy.y"));
}

TEST_CASE("constants take only compile time builtins") {
  CHECK_THROWS_AS(constant("D", dpdx(1.0)), ShaderError);
  CHECK_THROWS_AS(constant("I", inverse(mat2(1.0))), ShaderError);
  CHECK_NOTHROW(constant("S", sqrt(2.0) * consts::pi()));
}

TEST_CASE("non void functions return on every path") {
  CHECK_THROWS_AS(fn("a", {{"x", f32()}}, f32(), [](const Args& p) {
                    If(p["x"] > 0.0, [] { Return(1.0); });
                  }),
                  ShaderError);
  CHECK_THROWS_AS(fn("b", {}, f32(), [](const Args&) {
                    Loop([] {
                      Break();
                    });
                  }),
                  ShaderError);
  CHECK_NOTHROW(fn("c", {{"x", f32()}}, f32(), [](const Args& p) {
    If(p["x"] > 0.0, [] { Return(1.0); }).Else([] { Return(2.0); });
  }));
  CHECK_NOTHROW(fn("d", {{"x", i32()}}, f32(), [](const Args& p) {
    Switch(p["x"]).Case(0, [] { Return(0.0); }).Default([] { Return(1.0); });
  }));
  // an endless loop never reaches the end; a Break inside a nested loop does not leave it
  CHECK_NOTHROW(fn("e", {}, f32(), [](const Args&) {
    Loop([] {
      For(0, 2, [](const Val&) { Break(); });
      Return(1.0);
    });
  }));
  CHECK_NOTHROW(fn("f", {}, voidType(), [](const Args&) {}));
}

TEST_CASE("struct field names are unique before and after sanitizing") {
  CHECK_THROWS_AS(makeStruct("S", {{"a-b", f32()}, {"a_b", f32()}}), ShaderError);
  CHECK_THROWS_AS(makeStruct("S", {{"x", f32()}, {"x", vec2f()}}), ShaderError);
  CHECK_THROWS_AS(uniformBlock("u", {{"a.b", f32()}, {"a_b", f32()}}), ShaderError);
  CHECK_NOTHROW(makeStruct("S", {{"x", f32()}, {"y", f32()}}));
}

TEST_CASE("a uniform writer keeps its layout alive") {
  UniformWriter w(structLayout(makeStruct("Temp", {{"x", f32()}, {"y", vec4f()}})));
  w.setFloat("x", 2.0f);
  w.setVec4("y", 1, 2, 3, 4);
  CHECK(w.layout().structType->name == "Temp");
  CHECK(w.bytes().size() == 32);
}

TEST_CASE("a Break or Continue before the return can leave the block") {
  CHECK_THROWS_AS(fn("a", {{"x", i32()}}, f32(), [](const Args& p) {
                    Switch(p["x"]).Case(0, [] {
                      Break();
                      Return(1.0);
                    }).Default([] { Return(2.0); });
                  }),
                  ShaderError);
  CHECK_THROWS_AS(fn("b", {{"x", f32()}}, f32(), [](const Args& p) {
                    Loop([&] {
                      Switch(toI32(p["x"])).Default([] { Break(); });
                      If(p["x"] > 0.0, [] { Break(); });
                    });
                  }),
                  ShaderError);
  // a Continue in a nested switch leaves the switch case too
  CHECK_THROWS_AS(fn("c", {{"x", i32()}}, f32(), [](const Args& p) {
                    For(0, 2, [&](const Val&) {
                      Switch(p["x"]).Case(0, [] {
                        Switch(toI32(1)).Default([] { Continue(); });
                        Return(1.0);
                      }).Default([] { Return(2.0); });
                    });
                  }),
                  ShaderError);
  // a Break after the return is never reached
  CHECK_NOTHROW(fn("d", {{"x", i32()}}, f32(), [](const Args& p) {
    Switch(p["x"]).Case(0, [] { Return(1.0); }).Default([] { Return(2.0); });
  }));
}

TEST_CASE("a matrix argument only builds a matrix") {
  const Output out = output(0);
  program("m", "M", [] { Position(vec4(0.0)); }, [&] {
    const Val m4 = Let(mat4(1.0), "m4");
    const Val m3 = Let(mat3(1.0), "m3");
    CHECK_THROWS_AS(construct(vec3f(), {m3}), ShaderError);
    CHECK_THROWS_AS(vec4(m4), ShaderError);
    CHECK_THROWS_AS(mat4(m3), ShaderError);
    CHECK_THROWS_AS(mat3(m4, 1.0), ShaderError);
    CHECK(mat3(m4).type() == mat3f());
    CHECK(mat4(m4).type() == mat4f());
    out.set(vec4(1.0));
  });
}

TEST_CASE("function signatures take values, not atomics") {
  const TypePtr counter = makeStruct("Counter", {{"n", atomicU32()}});
  CHECK_THROWS_AS(fn("a", {{"x", atomicU32()}}, voidType(), [](const Args&) {}), ShaderError);
  CHECK_THROWS_AS(fn("b", {{"c", counter}}, voidType(), [](const Args&) {}), ShaderError);
  CHECK_THROWS_AS(fn("c", {}, atomicU32(), [](const Args&) { Return(1); }), ShaderError);
  CHECK_THROWS_AS(fn("d", {{"x", voidType()}}, voidType(), [](const Args&) {}), ShaderError);
}

TEST_CASE("a copied uniform writer writes its own buffer") {
  UniformWriter a(structLayout(makeStruct("C", {{"x", f32()}})));
  a.setFloat("x", 1.0f);
  UniformWriter b = a;
  b.setFloat("x", 2.0f);
  float av = 0, bv = 0;
  std::memcpy(&av, a.bytes().data(), 4);
  std::memcpy(&bv, b.bytes().data(), 4);
  CHECK(av == 1.0f);
  CHECK(bv == 2.0f);
  CHECK(a.bytes().data() != b.bytes().data());

  // a copy of a writer over an external buffer writes the same buffer
  std::array<std::byte, 16> external{};
  UniformWriter e(structLayout(makeStruct("E", {{"x", f32()}})), external);
  UniformWriter f = e;
  f.setFloat("x", 3.0f);
  float ev = 0;
  std::memcpy(&ev, external.data(), 4);
  CHECK(ev == 3.0f);

  // a moved writer keeps the bytes
  UniformWriter m = std::move(b);
  float mv = 0;
  std::memcpy(&mv, m.bytes().data(), 4);
  CHECK(mv == 2.0f);
}

TEST_CASE("a copied reflection keeps its layouts alive") {
  Reflection copy;
  {
    const UniformBlock ub = uniformBlock("cam", {{"m", mat4f()}, {"t", f32()}});
    const ProgramAST ast = program("r", "R", [&] { Position(ub["m"] * vec4(ub["t"])); }, [] {});
    copy = analyzeProgram(ast).reflection;
  }
  REQUIRE(copy.uniformBlocks.size() == 1);
  CHECK(copy.uniformBlocks[0].layout->find("t")->offset == 64);
  UniformWriter w(copy.uniformBlocks[0].layout);
  w.setFloat("t", 1.0f);
}

TEST_CASE("atomics need a read/write storage buffer") {
  CHECK_THROWS_AS(storage("ro", atomicU32(), StorageAccess::Read), ShaderError);
  CHECK_THROWS_AS(storage("ro", makeStruct("H", {{"n", atomicI32()}}), StorageAccess::Read),
                  ShaderError);
  CHECK_NOTHROW(storage("rw", atomicU32(), StorageAccess::ReadWrite));
}

TEST_CASE("texel offsets are in -8..7") {
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Output out = output(0);
  program("o", "O", [] { Position(vec4(0.0)); }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ivec2(100, 0)), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ivec2(-9, 0)), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ivec2(8)), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, -ivec2(9, 0)), ShaderError);
    CHECK_NOTHROW(albedo.sampleOffset(uv, ivec2(-8, 7)));
    out.set(vec4(1.0));
  });
}

TEST_CASE("WGSL emits no empty input struct") {
  const Output out = output(0);
  const ProgramAST render = program("e", "E", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); });
  const std::string wgsl = compileWGSL(analyzeProgram(render)).code;
  CHECK_FALSE(has(wgsl, "struct VSIn"));
  CHECK_FALSE(has(wgsl, "struct FSIn"));
  CHECK(has(wgsl, "@vertex fn vs_main() -> VSOut {"));
  CHECK(has(wgsl, "@fragment fn fs_main() -> FSOut {"));

  const StorageBuffer data = storage("data", f32());
  const ProgramAST compute = computeProgram("c", "C", {1, 1, 1}, [&] { data[0u].set(1.0); });
  const std::string cs = compileWGSL(analyzeProgram(compute)).code;
  CHECK_FALSE(has(cs, "struct CSIn"));
  CHECK(has(cs, "fn cs_main() {"));
}

TEST_CASE("stage inputs and outputs are numeric scalars or vectors") {
  CHECK_THROWS_AS(attribute("flag", boolType()), ShaderError);
  CHECK_THROWS_AS(attribute("m", mat4f()), ShaderError);
  CHECK_THROWS_AS(varying("b", vec2b(), Interpolation::Flat), ShaderError);
  CHECK_THROWS_AS(varying("s", makeStruct("V", {{"x", f32()}})), ShaderError);
  CHECK_THROWS_AS(output(0, boolType()), ShaderError);
  CHECK_THROWS_AS(output(0, mat2f()), ShaderError);
  CHECK_NOTHROW(attribute("id", u32()));
  CHECK_NOTHROW(output(1, vec4i()));
}

TEST_CASE("output indices are in 0..7 and one index has one type") {
  CHECK_THROWS_AS(output(-1), ShaderError);
  CHECK_THROWS_AS(output(8), ShaderError);
  const Output color = output(0, vec4f());
  const Output gray = output(0, f32());
  const Output again = output(0, vec4f());
  const ProgramAST clash = program("o", "O", [] { Position(vec4(0.0)); }, [&] {
    color.set(vec4(1.0));
    gray.set(1.0);
  });
  CHECK_THROWS_AS(analyzeProgram(clash), ShaderError);
  // two declarations of one index and type are one output
  const ProgramAST same = program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    color.set(vec4(1.0));
    again.set(vec4(0.5));
  });
  CHECK(analyzeProgram(same).reflection.outputs.size() == 1);
}

TEST_CASE("structs need a field") {
  CHECK_THROWS_AS(makeStruct("Empty", {}), ShaderError);
  CHECK_THROWS_AS(uniformBlock("empty", {}), ShaderError);
}

TEST_CASE("FragDepth and ClipDistance take f32") {
  CHECK_THROWS_AS(program("d", "D", [] { Position(vec4(0.0)); }, [] { FragDepth(toI32(1)); }),
                  ShaderError);
  CHECK_THROWS_AS(program("c", "C", [] { ClipDistance(0, toU32(1)); }, [] {}), ShaderError);
}

TEST_CASE("generated and GLSL reserved names are not usable") {
  const Output out = output(0);
  const auto use = [&](const Val& k) {
    return program("n", "N", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(k)); });
  };
  // main is a builtin name for a constant, so it is renamed, not rejected
  CHECK(has(compileGLSL(analyzeProgram(use(constant("main", 1.0)))).fragment, "const float main_ = 1.0;"));
  CHECK_THROWS_AS(analyzeProgram(use(constant("moga_anything", 1.0))), ShaderError);
  // gl_ and webgl_ prefixes are reserved by GLSL and WebGL, so they get a prefix
  const ProgramAST ast = program("g", "G", [] { Position(vec4(0.0)); }, [&] {
    const Val a = Let(1.0, "gl_x");
    const Val b = Let(2.0, "webgl_y");
    out.set(vec4(a + b + constant("gl_Z", 3.0)));
  });
  const AnalyzedProgram an = analyzeProgram(ast);
  const std::string glsl = compileGLSL(an).fragment;
  CHECK(has(glsl, "float xgl_x_"));
  CHECK(has(glsl, "float xwebgl_y_"));
  CHECK(has(glsl, "const float xgl_Z = 3.0;"));
}

TEST_CASE("unary minus needs a signed type") {
  const Output out = output(0);
  program("u", "U", [] { Position(vec4(0.0)); }, [&] {
    CHECK_THROWS_AS(-toU32(1), ShaderError);
    CHECK_THROWS_AS(-uvec2(1, 2), ShaderError);
    CHECK((-toI32(1)).type() == i32());
    CHECK((-mat2(1.0)).type() == mat2f());
    out.set(vec4(1.0));
  });
}

TEST_CASE("operands with side effects are evaluated once") {
  // a function that writes storage: calling it twice is visible
  const StorageBuffer counter = storage("counter", u32());
  const Function bump = fn("bump", {}, u32(), [&](const Args&) {
    counter[0u].increment();
    Return(counter[0u]);
  });
  const Function bumpF = fn("bumpF", {}, f32(), [&](const Args&) { Return(toF32(bump())); });
  const Output out = output(0, vec4u());
  const ProgramAST ast = program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val m = Let(mod(bumpF(), 3.0), "m");
    const Val d = Let(dot(uvec2(bump(), 1u), uvec2(1u, 2u)), "d");
    const Val sel = Let(select(bvec2(true, false), uvec2(bump(), 0u), uvec2(0u, 0u)), "sel");
    const Val sq = Let(pow2(bumpF()), "sq");
    const Val v = Var(uvec4(0u), "v");
    v[toI32(bump()) % 4].addAssign(1u);
    out.set(uvec4(toU32(m + sq) + d, sel.x(), sel.y(), v.x()));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "fn moga_mod_f32(a: f32, b: f32) -> f32"));
  CHECK(has(wgsl, "moga_mod_f32(f_bumpF(), 3.0f)"));
  // the compound assignment names its target once
  CHECK(has(wgsl, "] += 1u;"));
  // pow2 of a call goes through a constant
  CHECK(has(wgsl, "let once_"));
  CHECK_FALSE(has(wgsl, "f_bumpF() * f_bumpF()"));
}

TEST_CASE("GLSL helpers evaluate a call operand once") {
  const Function two = fn("two", {}, vec2i(), [](const Args&) { Return(ivec2(1, 2)); });
  const Table inst = table("inst", {{"model", mat4f()}}, ResourceFrequency::Object);
  const Function pick = fn("pick", {}, u32(), [](const Args&) { Return(0u); });
  const Output out = output(0, vec4i());
  const ProgramAST ast = program("g", "G", [&] { Position(inst.at(pick())["model"] * vec4(1.0)); }, [&] {
    const Val d = Let(dot(two(), two()), "d");
    const Val s = Let(select(bvec2(true, false), two(), ivec2(0, 0)), "s");
    out.set(ivec4(d, s, 0));
  });
  const CompiledGLSL glsl = compileGLSL(analyzeProgram(ast));
  CHECK(has(glsl.fragment, "int moga_dot_ivec2(ivec2 a, ivec2 b) { return a.x * b.x + a.y * b.y; }"));
  CHECK(has(glsl.fragment, "moga_dot_ivec2(f_two(), f_two())"));
  CHECK(has(glsl.fragment, "moga_select_ivec2(ivec2(0, 0), f_two(), bvec2(true, false))"));
  CHECK(has(glsl.vertex, "tb_inst_fetchMat4(f_pick(), 0)"));
}

TEST_CASE("select evaluates both values on every backend") {
  const Function pick = fn("pick", {}, f32(), [](const Args&) { Return(1.0); });
  const Output out = output(0);
  const ProgramAST ast = program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val c = Let(fragCoord().x() > 1.0, "c");
    const Val a = Let(select(c, pick(), 0.0), "a");
    const Val b = Let(select(c, 2.0, 0.0), "b");
    out.set(vec4(a + b));
  });
  const AnalyzedProgram an = analyzeProgram(ast);
  const std::string glsl = compileGLSL(an).fragment;
  // ?: would evaluate only one branch; the helper call evaluates both
  CHECK(has(glsl, "float moga_selectb_float(float f, float t, bool c) { return c ? t : f; }"));
  CHECK(has(glsl, "moga_selectb_float(0.0, f_pick(), c_"));
  // values without side effects keep ?:
  CHECK(has(glsl, "(c_"));
  CHECK(has(glsl, " ? 2.0 : 0.0)"));
  CHECK(has(compileWGSL(an).code, "select(0.0f, f_pick(), c_"));
  // select takes scalars and vectors only (WGSL)
  program("m", "M", [] { Position(vec4(0.0)); }, [&] {
    CHECK_THROWS_AS(select(toBool(true), mat2(1.0), mat2(0.0)), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("linear interpolation is WebGPU only") {
  const Varying uv = varying("uv", vec2f(), Interpolation::Linear);
  const Output out = output(0);
  const ProgramAST ast = program(
      "l", "L",
      [&] {
        uv.set(vec2(0.0));
        Position(vec4(0.0));
      },
      [&] { out.set(vec4(uv.value(), 0.0, 1.0)); });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(has(compileWGSL(a).code, "@interpolate(linear) v_uv: vec2f,"));
  CHECK_THROWS_WITH_AS(compileGLSL(a),
                       doctest::Contains("l: linear interpolation of varying uv needs WebGPU; WebGL2 cannot run it"),
                       ShaderError);
}

TEST_CASE("flat varyings say which vertex they take") {
  const Varying first = varying("first", u32(), Interpolation::Flat);
  const Varying any = varying("any", u32(), Interpolation::FlatEither);
  const Output out = output(0, vec4u());
  const auto build = [&](const Varying& v) {
    return program(
        "f", "F",
        [&] {
          v.set(vertexIndex());
          Position(vec4(0.0));
        },
        [&] { out.set(uvec4(v.value())); });
  };
  const ProgramAST astFirst = build(first);
  const AnalyzedProgram a = analyzeProgram(astFirst);
  CHECK(has(compileWGSL(a).code, "@interpolate(flat, first) v_first: u32,"));
  // WebGL2 takes the last vertex unless the renderer switches the convention
  CHECK_THROWS_AS(compileGLSL(a), ShaderError);
  CHECK(has(compileGLSL(a, {.firstVertexConvention = true}).vertex, "flat out uint v_first;"));

  const ProgramAST astAny = build(any);
  const AnalyzedProgram b = analyzeProgram(astAny);
  CHECK(has(compileWGSL(b).code, "@interpolate(flat, either) v_any: u32,"));
  CHECK(has(compileGLSL(b).vertex, "flat out uint v_any;"));
  CHECK_THROWS_AS(varying("id", u32(), Interpolation::Linear), ShaderError);
}

TEST_CASE("a Loop must be able to exit") {
  CHECK_THROWS_AS(fn("f", {}, f32(), [](const Args&) { Loop([] {}); }), ShaderError);
  CHECK_THROWS_AS(program("p", "P", [] { Loop([] {}); }, [] {}), ShaderError);
  // a Break in a nested loop leaves that loop, not this one
  CHECK_THROWS_AS(program("n", "N", [] {
    Loop([] { For(0, 2, [](const Val&) { Break(); }); });
  }, [] {}),
                  ShaderError);
  CHECK_NOTHROW(fn("g", {}, f32(), [](const Args&) { Loop([] { Return(1.0); }); }));
  CHECK_NOTHROW(fn("h", {{"x", f32()}}, f32(), [](const Args& p) {
    Loop([&] { If(p["x"] > 0.0, [] { Return(1.0); }); });
  }));
}

TEST_CASE("arrays of arrays are WebGPU only") {
  const Output out = output(0);
  const ProgramAST ast = program("a", "A", [] { Position(vec4(0.0)); }, [&] {
    const Val grid = Let(array({array({1.0, 2.0}), array({3.0, 4.0})}), "grid");
    out.set(vec4(grid[0][1]));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(has(compileWGSL(a).code, "array<array<f32, 2>, 2>"));
  CHECK_THROWS_WITH_AS(compileGLSL(a),
                       doctest::Contains("a: array of arrays array<array<f32, 2>, 2> needs WebGPU; WebGL2 cannot run it"),
                       ShaderError);
  // also through a struct field
  const TypePtr s = makeStruct("Grid", {{"cells", arrayOf(arrayOf(vec4f(), 2), 2)}});
  const ProgramAST viaStruct = program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val g = Var(s, "g");
    out.set(g.member("cells")[0][0]);
  });
  CHECK_THROWS_AS(compileGLSL(analyzeProgram(viaStruct)), ShaderError);
}

TEST_CASE("Else, ElseIf and Default are set once") {
  program("e", "E", [] { Position(vec4(0.0)); }, [] {
    const Val c = Let(fragCoord().x() > 1.0, "c");
    IfChain twice = If(c, [] {});
    twice.Else([] {});
    CHECK_THROWS_AS(twice.Else([] {}), ShaderError);
    IfChain chain = If(c, [] {});
    chain.Else([] {});
    CHECK_THROWS_AS(chain.ElseIf(c, [] {}), ShaderError);
    const Val i = Let(toI32(fragCoord().x()), "i");
    SwitchChain sw = Switch(i);
    sw.Default([] {});
    CHECK_THROWS_AS(sw.Default([] {}), ShaderError);
  });
}

TEST_CASE("depth cube textures, and no 3D depth textures") {
  const Texture shadowCube = texture("shadowCube", TextureDimension::Cube,
                                     {.sampleType = TextureSampleType::Depth, .sampled = true});
  CHECK_THROWS_AS(texture("d3", TextureDimension::D3, {.sampleType = TextureSampleType::Depth}),
                  ShaderError);
  const Output out = output(0);
  const ProgramAST ast = program("c", "C", [] { Position(vec4(0.0)); }, [&] {
    const Val d = Let(shadowCube.sample(vec3(1.0, 0.0, 0.0)), "d");
    out.set(vec4(d + toF32(shadowCube.size().x())));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "var t_shadowCube: texture_depth_cube;"));
  CHECK(has(wgsl, "textureSampleLevel(t_shadowCube, s_shadowCube, vec3f(1.0f, 0.0f, 0.0f), 0i)"));
  CHECK(has(compileGLSL(a).fragment, "uniform highp samplerCube t_shadowCube;"));
}

TEST_CASE("constants built on a runtime constant are functions too") {
  const Val copy = constant("COPY", consts::infinity());
  const Val twice = constant("TWICE", copy * 2.0);
  const Output out = output(0);
  const ProgramAST ast =
      program("k", "K", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(twice)); });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(wgsl, "fn COPY() -> f32 { return INFINITY(); }"));
  CHECK(has(wgsl, "fn TWICE() -> f32 { return (COPY() * 2.0f); }"));
  CHECK(has(wgsl, "vec4f(TWICE())"));
  // GLSL const expressions may use other constants and builtins
  CHECK(has(compileGLSL(a).fragment, "const float COPY = INFINITY;"));
  // a bitcast must be the whole value of a constant, of a literal
  CHECK_THROWS_AS(constant("X", uintBitsToFloat(toU32(0x7F800000u)) * 2.0), ShaderError);
  CHECK_THROWS_AS(constant("Y", uintBitsToFloat(toU32(1u) + toU32(2u))), ShaderError);
}

TEST_CASE("every GLSL and WGSL reserved word is renamed") {
  const TypePtr s = makeStruct("R", {{"attribute", f32()}, {"varying", f32()}, {"noperspective", f32()}});
  const Output out = output(0);
  const ProgramAST ast = program("r", "R", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Var(s, "v");
    out.set(vec4(v.member("attribute") + constant("sin", 1.0) + constant("handle", 2.0) +
                 constant("fallthrough", 3.0)));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  const std::string wgsl = compileWGSL(a).code;
  CHECK(has(glsl, "float attribute_;"));
  CHECK(has(glsl, "float varying_;"));
  CHECK(has(glsl, "float noperspective_;"));
  // a constant cannot take a builtin function name; other names stay as they are
  CHECK(has(glsl, "const float sin_ = 1.0;"));
  CHECK(has(glsl, "const float handle = 2.0;"));
  // a WGSL reserved word
  CHECK(has(wgsl, "const fallthrough_: f32 = 3.0f;"));
}

TEST_CASE("a swizzle write forms its target before its value") {
  const StorageBuffer data = storage("data", vec4f());
  const Function where = fn("pickIndex", {}, u32(), [](const Args&) { Return(0u); });
  const Function what = fn("pickValue", {}, vec2f(), [](const Args&) { Return(vec2(1.0)); });
  const ProgramAST ast = computeProgram("w", "W", {1, 1, 1}, [&] { data[where()].swizzle("zx").set(what()); });
  const std::string wgsl = compileWGSL(analyzeProgram(ast)).code;
  const auto target = wgsl.find("let moga_target = &sb_data[f_pickIndex()];");
  const auto value = wgsl.find("let moga_swizzle = f_pickValue();");
  REQUIRE(target != std::string::npos);
  REQUIRE(value != std::string::npos);
  CHECK(target < value);
}

TEST_CASE("texel offsets evaluate ~ like the shader") {
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Output out = output(0);
  program("o", "O", [] { Position(vec4(0.0)); }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    // -~(8, 0) is (9, 1) in the shader
    CHECK_THROWS_AS(albedo.sampleOffset(uv, -~ivec2(8, 0)), ShaderError);
    // ~(-8, 0) is (7, -1), inside -8..7
    CHECK_NOTHROW(albedo.sampleOffset(uv, ~ivec2(-8, 0)));
    // ~7 is -8, the lower bound
    CHECK_NOTHROW(albedo.sampleOffset(uv, ~ivec2(7)));
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ~ivec2(-10, 0)), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("constant subgroup ids are in 0..127") {
  computeProgram("g", "G", {32, 1, 1}, [] {
    const Val v = Let(toF32(subgroupInvocationId()), "v");
    CHECK_THROWS_AS(subgroupBroadcast(v, 128), ShaderError);
    CHECK_THROWS_AS(subgroupShuffle(v, 128), ShaderError);
    CHECK_THROWS_AS(subgroupBroadcast(v, -1), ShaderError);
    CHECK_NOTHROW(subgroupBroadcast(v, 127));
    CHECK_NOTHROW(subgroupShuffle(v, 0));
    // a computed id is checked by the shader, not when it is built
    CHECK_NOTHROW(subgroupShuffle(v, localIndex() + 200u));
  });
}

TEST_CASE("named constant subgroup ids are checked like literals") {
  const Val tooFar = constant("TOO_FAR", toU32(128));
  const Val lane = constant("LANE", toU32(3));
  computeProgram("c", "C", {32, 1, 1}, [&] {
    const Val v = Let(toF32(subgroupInvocationId()), "v");
    CHECK_THROWS_WITH_AS(subgroupShuffle(v, tooFar), doctest::Contains("subgroupShuffle: the id 128 is not in 0..127"), ShaderError);
    CHECK_THROWS_WITH_AS(subgroupBroadcast(v, tooFar), doctest::Contains("subgroupBroadcast: the id 128 is not in 0..127"), ShaderError);
    // WGSL takes any const expression as the broadcast id
    CHECK_NOTHROW(subgroupBroadcast(v, lane));
    CHECK_NOTHROW(subgroupBroadcast(v, lane * 2u + 1u));
    CHECK_THROWS_AS(subgroupBroadcast(v, lane * 64u), ShaderError);
  });
}

TEST_CASE("the vertex stage writes Position() on every path") {
  const Output out = output(0);
  const auto fragment = [&] { out.set(vec4(1.0)); };
  const auto analyze = [&](const std::function<void()>& vertex) {
    const ProgramAST ast = program("p", "P", vertex, fragment);
    return analyzeProgram(ast);
  };
  CHECK_THROWS_AS(analyze([] {}), ShaderError);
  // only one branch writes
  CHECK_THROWS_AS(analyze([] { If(toBool(true), [] { Position(vec4(0.0)); }); }), ShaderError);
  // a Return before the write
  CHECK_THROWS_AS(analyze([] {
    If(toBool(true), [] { Return(); });
    Position(vec4(0.0));
  }),
                  ShaderError);
  // a For may run zero times
  CHECK_THROWS_AS(analyze([] { For(0, 1, [](const Val&) { Position(vec4(0.0)); }); }), ShaderError);
  // a switch without a default
  CHECK_THROWS_AS(analyze([] { Switch(toI32(vertexIndex())).Case(0, [] { Position(vec4(0.0)); }); }),
                  ShaderError);

  CHECK_NOTHROW(analyze([] { Position(vec4(0.0)); }));
  CHECK_NOTHROW(analyze([] {
    If(toBool(true), [] { Position(vec4(1.0)); }).Else([] { Position(vec4(0.0)); });
  }));
  // a write, then an early Return
  CHECK_NOTHROW(analyze([] {
    Position(vec4(0.0));
    If(toBool(true), [] { Return(); });
  }));
  // a Loop body runs at least once
  CHECK_NOTHROW(analyze([] {
    Loop([] {
      Position(vec4(0.0));
      Break();
    });
  }));
  CHECK_NOTHROW(analyze([] {
    Switch(toI32(vertexIndex())).Case(0, [] { Position(vec4(1.0)); }).Default([] { Position(vec4(0.0)); });
  }));
  // a compute program has no vertex stage
  CHECK_NOTHROW(analyzeProgram(computeProgram("c", "C", {1, 1, 1}, [] {})));
}

TEST_CASE("a Break that skips Position() is a path without a write") {
  const Output out = output(0);
  const auto analyze = [&](const std::function<void()>& vertex) {
    const ProgramAST ast = program("p", "P", vertex, [&] { out.set(vec4(1.0)); });
    return analyzeProgram(ast);
  };
  CHECK_THROWS_AS(analyze([] {
    Loop([] {
      If(toBool(true), [] { Break(); });
      Position(vec4(0.0));
    });
  }),
                  ShaderError);
  // a write before the conditional Break covers both ways out
  CHECK_NOTHROW(analyze([] {
    Loop([] {
      Position(vec4(0.0));
      If(toBool(true), [] { Break(); });
      Break();
    });
  }));
  // a Break in a switch case leaves only the switch; the write after it covers the path
  CHECK_NOTHROW(analyze([] {
    Switch(toI32(vertexIndex())).Case(0, [] { Break(); }).Default([] {});
    Position(vec4(0.0));
  }));
  // a Continue repeats the loop; the exit is still after the write
  CHECK_NOTHROW(analyze([] {
    Loop([] {
      If(toBool(false), [] { Continue(); });
      Position(vec4(0.0));
      Break();
    });
  }));
}

TEST_CASE("locals and values are never void or atomic") {
  const TypePtr counter = makeStruct("Counter", {{"n", atomicU32()}, {"x", f32()}});
  const StorageBuffer counts = storage("counts", atomicU32());
  const StorageBuffer records = storage("records", counter);
  computeProgram("v", "V", {1, 1, 1}, [&] {
    CHECK_THROWS_AS(Var(atomicU32(), "a"), ShaderError);
    CHECK_THROWS_AS(Var(voidType(), "b"), ShaderError);
    CHECK_THROWS_AS(Var(counter, "c"), ShaderError);
    CHECK_THROWS_AS(Var(arrayOf(atomicI32(), 2), "d"), ShaderError);
    // a value read from storage cannot be copied into a local either
    CHECK_THROWS_AS(Let(counts[0u], "e"), ShaderError);
    CHECK_THROWS_AS(Var(records[0u], "f"), ShaderError);
    CHECK_THROWS_AS(construct(counter, {0u, 1.0}), ShaderError);
    CHECK_THROWS_AS(records[0u].set(records[1u]), ShaderError);
    // the plain field of a struct with an atomic is still a value
    CHECK(Let(records[0u].member("x"), "g").type() == f32());
  });
}

TEST_CASE("void cannot be nested in an array or a struct") {
  CHECK_THROWS_AS(arrayOf(voidType(), 2), ShaderError);
  CHECK_THROWS_AS(makeStruct("V", {{"x", f32()}, {"nothing", voidType()}}), ShaderError);
  CHECK_THROWS_AS(uniformBlock("u", {{"nothing", voidType()}}), ShaderError);
  // atomics stay valid as array elements, for storage and workgroup memory
  CHECK_NOTHROW(arrayOf(atomicU32(), 2));
}

TEST_CASE("matrix constructors take only the forms both backends share") {
  const Output out = output(0);
  const ProgramAST ast = program("m", "M", [] { Position(vec4(0.0)); }, [&] {
    const Val c = Let(vec3(1.0), "c");
    const Val d = Let(mat3(2.0), "d");
    const Val cols = Let(mat3(c, c, c), "cols");
    const Val comps = Let(mat2(1.0, 0.0, 0.0, 1.0), "comps");
    CHECK_THROWS_AS(mat2(vec4(1.0)), ShaderError);
    CHECK_THROWS_AS(mat2(vec2(1.0), 1.0, 0.0), ShaderError);
    CHECK_THROWS_AS(mat3(c, c), ShaderError);
    CHECK_THROWS_AS(mat2(toI32(1)), ShaderError);
    out.set(vec4(d * cols * c, comps[0][0]));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  // one scalar is a diagonal matrix: native in GLSL, written out in WGSL
  CHECK(has(compileGLSL(a).fragment, "mat3 d_"));
  CHECK(has(compileGLSL(a).fragment, " = mat3(2.0);"));
  CHECK(has(compileWGSL(a).code, "mat3x3f(2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f)"));
}

TEST_CASE("an analyzed program owns its AST") {
  const Output out = output(0);
  // the program() result is a temporary; the analysis must not keep a pointer to it
  const AnalyzedProgram a =
      analyzeProgram(program("t", "T", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); }));
  CHECK(has(compileWGSL(a).code, "fout.o_0 = vec4f(1.0f);"));
  CHECK(has(compileGLSL(a).fragment, "o_0 = vec4(1.0);"));
}

TEST_CASE("vector parts have the target component kind") {
  const Output out = output(0);
  program("v", "V", [] { Position(vec4(0.0)); }, [&] {
    const Val i2 = Let(ivec2(1, 2), "i2");
    CHECK_THROWS_AS(vec3(i2, 2.0), ShaderError);
    CHECK_THROWS_AS(vec3(toI32(1)), ShaderError);
    CHECK_THROWS_AS(ivec4(i2, vec2(1.0)), ShaderError);
    // one value of the same size converts its kind
    CHECK(vec2(i2).type() == vec2f());
    CHECK(bvec2(vec2(1.0)).type() == vec2b());
    // convert() makes a scalar the target kind before it fills the vector
    CHECK(convert(toI32(3), vec3f()).type() == vec3f());
    out.set(vec4(vec2(i2), toVec2(toI32(1))));
  });
}

TEST_CASE("For steps move the counter toward the end") {
  const Output out = output(0);
  program("f", "F", [] { Position(vec4(0.0)); }, [&] {
    CHECK_THROWS_AS(For(0, 10, [](const Val&) {}, 0), ShaderError);
    CHECK_THROWS_AS(For(0, 10, [](const Val&) {}, -1), ShaderError);
    CHECK_THROWS_AS(For({.start = 10, .end = 0, .step = 1, .condition = BinaryOp::Gt}, [](const Val&) {}),
                    ShaderError);
    CHECK_NOTHROW(For({.start = 10, .end = 0, .step = -2, .condition = BinaryOp::Ge}, [](const Val&) {}));
    CHECK_NOTHROW(For(0, 10, [](const Val&) {}, 3));
    out.set(vec4(1.0));
  });
}

TEST_CASE("GLSL starts stage outputs at zero, as WGSL does") {
  const Varying uv = varying("uv", vec2f());
  const Output color = output(0, vec4f());
  const Val flag = uniform("flag", u32());
  const ProgramAST ast = program(
      "z", "Z",
      [&] {
        If(flag.eq(1u), [&] { uv.set(vec2(1.0)); });
        Position(vec4(0.0));
      },
      [&] { If(uv.value().x() > 0.5, [&] { color.set(vec4(1.0)); }); });
  const CompiledGLSL g = compileGLSL(analyzeProgram(ast));
  // the zero write comes before the branch that may skip the real one
  const std::size_t zeroUv = g.vertex.find("v_uv = vec2(0);");
  REQUIRE(zeroUv != std::string::npos);
  CHECK(zeroUv < g.vertex.find("if ("));
  const std::size_t zeroColor = g.fragment.find("o_0 = vec4(0);");
  REQUIRE(zeroColor != std::string::npos);
  CHECK(zeroColor < g.fragment.find("if ("));
}

TEST_CASE("clamp rejects constant bounds in the wrong order") {
  const Output out = output(0);
  const Val top = constant("clampTop", 0.25);
  program("c", "C", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(vec3(0.5), "x");
    const Val runtime = Let(0.9, "runtime");
    CHECK_THROWS_WITH_AS(clamp(0.5, 1.0, 0.0), doctest::Contains("clamp: the low bound 1 is above the high bound 0"),
                         ShaderError);
    CHECK_THROWS_WITH_AS(clamp(x, vec3(0.0, 2.0, 0.0), vec3(1.0)),
                         doctest::Contains("clamp: the low bound 2 is above the high bound 1 in component 1"), ShaderError);
    CHECK_THROWS_AS(clamp(x, 0.5, top), ShaderError);
    CHECK_THROWS_AS(clamp(x, -(-1.0 * vec3(1.0)), 0.0), ShaderError);
    CHECK_THROWS_AS(clamp(ivec2(5, 6), 3, 1), ShaderError);
    CHECK_THROWS_AS(clamp(x, vec3(4.0) / 2.0, 1.5), ShaderError);
    // equal bounds are valid; a bound known only at run time is not checked
    CHECK(clamp(x, 0.5, 0.5).type() == vec3f());
    CHECK(clamp(x, 1.0, runtime).type() == vec3f());
    out.set(vec4(1.0));
  });
}

TEST_CASE("smoothstep rejects equal constant edges") {
  const Output out = output(0);
  const Val edge = constant("smoothEdge", 1.0);
  program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(vec2(0.5), "x");
    CHECK_THROWS_WITH_AS(smoothstep(1.0, 1.0, 0.5), doctest::Contains("smoothstep: both edges are 1"), ShaderError);
    CHECK_THROWS_WITH_AS(smoothstep(vec2(0.0, 2.0), vec2(1.0, 2.0), x),
                         doctest::Contains("smoothstep: both edges are 2 in component 1"), ShaderError);
    CHECK_THROWS_AS(smoothstep(edge, 2.0 - 1.0, x), ShaderError);
    // reversed edges are valid in WGSL
    CHECK(smoothstep(1.0, 0.0, x).type() == vec2f());
    out.set(vec4(1.0));
  });
}

TEST_CASE("GLSL smoothstep follows the WGSL formula unless the edges are known in order") {
  const Output out = output(0);
  const ProgramAST ast = program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(vec2(0.5), "x");
    const Val edge = Let(0.25, "edge");
    const Val ordered = Let(smoothstep(0.0, 1.0, x), "ordered");
    const Val reversed = Let(smoothstep(1.0, 0.0, x), "reversed");
    const Val runtime = Let(smoothstep(edge, 1.0, x), "runtime");
    out.set(vec4(ordered + reversed, runtime));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string glsl = compileGLSL(a).fragment;
  // GLSL ES 3.00 8.3: undefined when edge0 >= edge1; WGSL 17.5: clamp((x - low) / (high - low), 0, 1), then t * t * (3 - 2 * t)
  CHECK(has(glsl, "vec2 moga_smoothstep_vec2(vec2 e0, vec2 e1, vec2 x) { vec2 t = clamp((x - e0) / (e1 - e0), 0.0, "
                  "1.0); return t * t * (3.0 - 2.0 * t); }"));
  CHECK(has(glsl, "smoothstep(vec2(0.0), vec2(1.0), "));
  CHECK(has(glsl, "moga_smoothstep_vec2(vec2(1.0), vec2(0.0), "));
  // an edge known only at run time may be in either order
  CHECK(has(glsl, "moga_smoothstep_vec2(vec2(edge_"));
  const std::string wgsl = compileWGSL(a).code;
  CHECK_FALSE(has(wgsl, "moga_smoothstep"));
}

TEST_CASE("constant checks compare values as the shader stores them") {
  const Output out = output(0);
  const Val one = constant("unitEdge", 1.0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(0.5, "x");
    // 1 + 1e-8 is not an f32: WGSL may store 1 or the next f32 (1 + 2^-23), so the edges can be equal
    CHECK_THROWS_WITH_AS(smoothstep(1.0, 1.0 + 1e-8, x), doctest::Contains("smoothstep: the edges 1 and 1 to 1.0000001192092896 can be equal"),
                         ShaderError);
    CHECK_THROWS_AS(smoothstep(one, one + 1e-8, x), ShaderError);
    // and the low bound can be 1 + 2^-23, above the high bound 1
    CHECK_THROWS_WITH_AS(clamp(x, 1.0 + 1e-8, 1.0), doctest::Contains("can be above the high bound 1"), ShaderError);
    CHECK_THROWS_AS(clamp(x, one + 1e-8, one), ShaderError);
    // one f32 step above 1 (2^-23) stays distinct
    CHECK(smoothstep(1.0, 1.0 + 1.0 / 8388608.0, x).type() == f32());
    out.set(vec4(1.0));
  });
}

TEST_CASE("constant checks wrap i32 and u32 arithmetic modulo 2^32") {
  const Output out = output(0, vec4u());
  program("w", "W", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(toU32(5.0), "x");
    const Val xi = Let(toI32(5.0), "xi");
    const Val maxU = toU32(4294967295.0);
    // 4294967295u + 1u is 0u, so this low bound is below the high one
    CHECK(clamp(x, maxU + toU32(1), 3u).type() == u32());
    CHECK_THROWS_WITH_AS(clamp(x, 1u, maxU + toU32(1)), doctest::Contains("clamp: the low bound 1 is above the high bound 0"),
                         ShaderError);
    // 0u - 1u is 4294967295u
    CHECK_THROWS_WITH_AS(clamp(x, toU32(0) - toU32(1), 3u),
                         doctest::Contains("clamp: the low bound 4294967295 is above the high bound 3"), ShaderError);
    // 2147483647i + 1i is -2147483648i
    CHECK_THROWS_WITH_AS(clamp(xi, 0, toI32(2147483647.0) + toI32(1)),
                         doctest::Contains("clamp: the low bound 0 is above the high bound -2147483648"), ShaderError);
    // (2^32 - 1)^2 = 2^64 - 2^33 + 1, which is 1 modulo 2^32; a double would lose the low bits
    CHECK_THROWS_WITH_AS(clamp(x, 2u, maxU * maxU), doctest::Contains("clamp: the low bound 2 is above the high bound 1"),
                         ShaderError);
    out.set(uvec4(x));
  });
}

TEST_CASE("For checks steps known at build time, not only number literals") {
  const Output out = output(0);
  const Val zero = constant("zeroStep", 0.0);
  program("f", "F", [] { Position(vec4(0.0)); }, [&] {
    const auto body = [](const Val&) {};
    CHECK_THROWS_WITH_AS(For(0, 10, body, toI32(0)), doctest::Contains("For: a step of 0 never reaches the end with \"<\""),
                         ShaderError);
    CHECK_THROWS_WITH_AS(For(0, 10, body, toI32(-1)), doctest::Contains("For: a step of -1 never reaches the end with \"<\""),
                         ShaderError);
    CHECK_THROWS_AS(For(0, 10, body, toI32(zero)), ShaderError);
    CHECK_THROWS_AS(For(0, 10, body, toI32(2) - toI32(2)), ShaderError);
    CHECK_THROWS_AS(For(ForRange{toI32(10), 0, toI32(1), BinaryOp::Gt}, body), ShaderError);
    // 0u - 1u wraps to 4294967295u: it moves a u32 counter up, the wrong way for ">"
    CHECK_THROWS_AS(For(ForRange{toU32(10), toU32(0), toU32(0) - toU32(1), BinaryOp::Gt}, body), ShaderError);
    // f32 2.9 truncates to the step 2
    CHECK_NOTHROW(For(0, 10, body, toI32(toF32(2.9))));
    CHECK_NOTHROW(For(ForRange{toI32(10), 0, toI32(-2), BinaryOp::Gt}, body));
    out.set(vec4(1.0));
  });
}

TEST_CASE("constant checks read swizzles of known values") {
  const Output out = output(0);
  const Val lo = constant("LO", vec2(2.0, 0.0));
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Val shifts = constant("SHIFTS", ivec4(9, 1, -2, 0));
  program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(0.5, "x");
    CHECK_THROWS_WITH_AS(clamp(x, lo.x(), 1.0), doctest::Contains("clamp: the low bound 2 is above the high bound 1"), ShaderError);
    CHECK(clamp(x, lo.y(), 1.0).type() == f32());
    // (0, 2) against (1, 1): component 1 is reversed
    CHECK_THROWS_AS(clamp(vec2(0.5), lo.swizzle("yx"), vec2(1.0)), ShaderError);
    const Val uv = Let(vec2(0.5), "uv");
    // SHIFTS.yz is (1, -2), in -8..7; SHIFTS.xy holds 9
    CHECK_NOTHROW(albedo.sampleOffset(uv, shifts.swizzle("yz")));
    CHECK_THROWS_AS(albedo.sampleOffset(uv, shifts.swizzle("xy")), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("const expression checks are separate from the values the DSL can compute") {
  // WGSL evaluates min() at shader creation; the DSL computes it exactly
  const Val tooFar = constant("FAR_ID", min(toU32(128), toU32(129)));
  const Val lane = constant("NEAR_ID", min(toU32(3), toU32(9)));
  // sin() is a const expression whose value the DSL does not compute
  const Val wave = constant("WAVE_ID", toU32(floor(sin(toF32(1.0)) * 4.0)));
  // a bitcast constant is emitted as a WGSL function, so it is not a const expression
  const Val bits = constant("BITS_ID", floatBitsToUint(toF32(1.0)));
  computeProgram("k", "K", {32, 1, 1}, [&] {
    const Val v = Let(toF32(subgroupInvocationId()), "v");
    CHECK_THROWS_WITH_AS(subgroupShuffle(v, tooFar), doctest::Contains("subgroupShuffle: the id 128 is not in 0..127"), ShaderError);
    CHECK_THROWS_AS(subgroupBroadcast(v, tooFar), ShaderError);
    CHECK_NOTHROW(subgroupBroadcast(v, lane));
    CHECK_NOTHROW(subgroupBroadcast(v, wave));
    CHECK_THROWS_WITH_AS(subgroupBroadcast(v, bits), doctest::Contains("subgroupBroadcast: the id must be a constant"), ShaderError);
  });
  const Output out = output(0);
  const Texture albedo = texture("albedo", TextureDimension::D2);
  program("o", "O", [] { Position(vec4(0.0)); }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    const Val x = Let(0.5, "x");
    // max(2, 1) is 2, above the high bound 1
    CHECK_THROWS_WITH_AS(clamp(x, max(2.0, 1.0), 1.0), doctest::Contains("clamp: the low bound 2 is above the high bound 1"), ShaderError);
    CHECK_NOTHROW(albedo.sampleOffset(uv, min(ivec2(9, 0), ivec2(7, 7))));
    CHECK_THROWS_AS(albedo.sampleOffset(uv, max(ivec2(8, 0), ivec2(0, 0))), ShaderError);
    // round() of a halfway value is even, as in WGSL: 0.5 gives 0, so the step is 0, not 1
    CHECK_THROWS_AS(For(0, 10, [](const Val&) {}, toI32(round(toF32(0.5)))), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("constant checks give the same result in every host rounding mode") {
  struct Downward {
    int previous = std::fegetround();
    Downward() { std::fesetround(FE_DOWNWARD); }
    ~Downward() { std::fesetround(previous); }
  } downward;
  const Output out = output(0);
  computeProgram("r", "R", {32, 1, 1}, [&] {
    const Val v = Let(toF32(subgroupInvocationId()), "v");
    // WGSL round() is half to even: 127.5 is 128, out of range; 126.5 is 126
    CHECK_THROWS_WITH_AS(subgroupShuffle(v, toU32(round(toF32(127.5)))), doctest::Contains("subgroupShuffle: the id 128 is not in 0..127"),
                         ShaderError);
    CHECK_NOTHROW(subgroupBroadcast(v, toU32(round(toF32(126.5)))));
  });
  program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(0.5, "x");
    // 1 + 0.75 * 2^-23 is between 1 and 1 + 2^-23; WGSL may store either, so the edges can be equal
    CHECK_THROWS_AS(smoothstep(1.0, 1.0 + 0.75 / 8388608.0, x), ShaderError);
    out.set(vec4(1.0));
  });
  // f16 uniform bytes too: 1 + 3 * 2^-11 is halfway between 1 + 2^-10 and 1 + 2^-9; to even is 1 + 2^-9, bits 0x3c02
  UniformWriter w(structLayout(makeStruct("H", {{"h", f16()}})));
  w.setFloat("h", 1.0f + 3.0f / 2048.0f);
  std::uint16_t bits;
  std::memcpy(&bits, w.bytes().data(), 2);
  CHECK(bits == 0x3c02);
  CHECK(std::fegetround() == FE_DOWNWARD);
}

TEST_CASE("integer bit operations are evaluated, and invalid constant shifts are rejected") {
  const Output out = output(0);
  const Texture albedo = texture("albedo", TextureDimension::D2);
  program("b", "B", [&] {
    // Tint (Chrome 155) rejects each of these when it creates the shader
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Shl, toI32(1), toU32(31)), doctest::Contains("1 << 31 does not fit in i32"), ShaderError);
    CHECK_THROWS_AS(binary(BinaryOp::Shl, toI32(-2), toU32(31)), ShaderError);
    CHECK_THROWS_AS(binary(BinaryOp::Shl, toU32(3), toU32(31)), ShaderError);
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Shl, toI32(1), toU32(32)), doctest::Contains("Shift by 32 is not in 0..31"), ShaderError);
    // an i32 amount becomes u32 in WGSL: -1 is 4294967295
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Shl, toI32(1), toI32(-1)), doctest::Contains("Shift by -1 is not in 0..31"), ShaderError);
    CHECK_THROWS_AS(binary(BinaryOp::Shr, toU32(8), toI32(-3)), ShaderError);
    CHECK_NOTHROW(binary(BinaryOp::Shl, toI32(1), toI32(3)));
    CHECK_THROWS_AS(binary(BinaryOp::Shr, toU32(1), toU32(32)), ShaderError);
    // and accepts these
    CHECK_NOTHROW(binary(BinaryOp::Shl, toI32(-1), toU32(31)));
    CHECK_NOTHROW(binary(BinaryOp::Shl, toU32(1), toU32(31)));
    CHECK_NOTHROW(binary(BinaryOp::Shl, toI32(vertexIndex()), toU32(31)));
    Position(vec4(0.0));
  }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    // 1 << 4 is 16, outside -8..7
    CHECK_THROWS_WITH_AS(albedo.sampleOffset(uv, ivec2(binary(BinaryOp::Shl, toI32(1), toU32(4)), 0)),
                         doctest::Contains("albedo: texel offset component 16 is not in -8..7"), ShaderError);
    // 12 & 10 is 8
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ivec2(binary(BinaryOp::BitAnd, toI32(12), toI32(10)), 0)), ShaderError);
    // -16 >> 2 is -4 (arithmetic), -1 ^ 5 is -6, 1 << 2 is 4, 7 % 5 is 2
    CHECK_NOTHROW(albedo.sampleOffset(uv, ivec2(binary(BinaryOp::Shr, toI32(-16), toU32(2)),
                                                binary(BinaryOp::BitXor, toI32(-1), toI32(5)))));
    CHECK_NOTHROW(albedo.sampleOffset(uv, ivec2(binary(BinaryOp::Shl, toI32(1), toU32(2)), toI32(7) % toI32(5))));
    // a constant the DSL cannot evaluate cannot be range checked, so it is rejected
    CHECK_THROWS_WITH_AS(albedo.sampleOffset(uv, ivec2(toI32(floor(sin(toF32(1.0)) * 4.0)), 0)),
                         doctest::Contains("albedo: the texel offset must be a constant the DSL can evaluate (literals, named constants, "
                         "arithmetic, bit operations, min, max, clamp, abs, constructors)"),
                         ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("a value reused many times is analyzed and printed once") {
  const Function bump = fn("bump", {}, f32(), [](const Args&) { Return(1.0); });
  const Output out = output(0);
  const ProgramAST ast = program("d", "D", [] { Position(vec4(0.0)); }, [&] {
    Val x = Let(fragCoord().x(), "x");
    // 30 doublings: 31 nodes, but 2^30 leaves when written out as a tree
    for (int i = 0; i < 30; ++i) x = x + x;
    const Val y = Let(x, "y");
    // the same value again in a later statement and inside a nested block
    const Val z = Var(x * 2.0, "z");
    If(x > 2.0, [&] { z.set(z + x); });
    // a call is not shared: each use runs it, as the program says
    const Val twice = Let(bump() + bump(), "twice");
    const Val calls = Let(construct(f32(), {twice}), "calls");
    out.set(vec4(y, calls, 0.0, 1.0));
  });
  const auto start = std::chrono::steady_clock::now();
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string w = compileWGSL(a).code;
  const std::string g = compileGLSL(a).fragment;
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  CHECK(seconds < 1.0);
  // each doubling is one line; a tree would be gigabytes
  CHECK(w.size() < 20000);
  CHECK(g.size() < 20000);
  CHECK(has(w, "f_bump() + f_bump()"));
  // every local is declared once; WebGL2 and Tint reject a redeclaration in one scope
  const auto declaredOnce = [](const std::string& code, std::string_view head) {
    std::map<std::string, int> count;
    for (std::size_t at = code.find(head); at != std::string::npos; at = code.find(head, at + 1)) {
      const std::size_t start = code.find("moga_shared_", at);
      const std::size_t end = code.find_first_of(":= ", start);
      ++count[code.substr(start, end - start)];
    }
    return !count.empty() && std::ranges::all_of(count, [](const auto& c) { return c.second == 1; });
  };
  CHECK(declaredOnce(w, "let moga_shared_"));
  CHECK(declaredOnce(g, "float moga_shared_"));
}

TEST_CASE("only varyings that the fragment stage reads cross the stage boundary") {
  const Function bump = fn("bumpVarying", {}, f32(), [](const Args&) { Return(1.0); });
  const Varying used = varying("used", f32());
  const Varying unused = varying("unused", f32());
  const Varying called = varying("called", f32());
  const Output out = output(0);
  const ProgramAST ast = program("v", "V", [&] {
    used.set(1.0);
    unused.set(2.0);
    called.set(bump());
    Position(vec4(0.0));
  }, [&] { out.set(vec4(used.value())); });
  const AnalyzedProgram a = analyzeProgram(ast);
  REQUIRE(a.reflection.varyings.size() == 1);
  CHECK(a.reflection.varyings[0].name == "used");
  const std::string w = compileWGSL(a).code;
  CHECK(has(w, "v_used"));
  CHECK_FALSE(has(w, "v_unused"));
  CHECK_FALSE(has(w, "v_called"));
  // the call in a dropped varying's value still runs
  CHECK(has(w, "_ = f_bumpVarying();"));
  const CompiledGLSL g = compileGLSL(a);
  CHECK_FALSE(has(g.vertex, "v_unused"));
  CHECK(has(g.vertex, "f_bumpVarying();"));
}

TEST_CASE("difference of unsigned values subtracts the smaller from the larger") {
  const Output out = output(0, vec4u());
  const ProgramAST ast = program("d", "D", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(toU32(fragCoord().x()), "x");
    const Val y = Let(toU32(fragCoord().y()), "y");
    out.set(uvec4(difference(x, y), 0u, 0u, 0u));
  });
  const std::string w = compileWGSL(analyzeProgram(ast)).code;
  // |x - y| for u32: abs() of a wrapped u32 subtraction would give 4294967295 for 1 and 2
  CHECK(has(w, "(max(x_"));
  CHECK(has(w, ") - min(x_"));
  CHECK_FALSE(has(w, "abs("));
}

TEST_CASE("For rejects a counter that wraps around before it reaches the end") {
  const Output out = output(0);
  program("f", "F", [] { Position(vec4(0.0)); }, [&] {
    const auto body = [](const Val&) {};
    // 4294967294, then 4294967294 + 2 wraps to 0, which is below the end again
    CHECK_THROWS_WITH_AS(For(ForRange{toU32(4294967294.0), toU32(4294967295.0), toU32(2), BinaryOp::Lt}, body),
                         doctest::Contains("For: the counter wraps around after 4294967294 instead of reaching the end 4294967295"),
                         ShaderError);
    // every u32 is <= 4294967295, so the counter wraps
    CHECK_THROWS_AS(For(ForRange{toU32(4294967290.0), toU32(4294967295.0), toU32(1), BinaryOp::Le}, body), ShaderError);
    // counting down past the smallest i32
    CHECK_THROWS_AS(For(ForRange{toI32(-2147483647), toI32(-2147483647) - toI32(1), toI32(-2), BinaryOp::Ge}, body),
                    ShaderError);
    // reaching the end exactly, or stopping before the limit, is fine
    CHECK_NOTHROW(For(ForRange{toU32(4294967294.0), toU32(4294967295.0), toU32(1), BinaryOp::Lt}, body));
    CHECK_NOTHROW(For(ForRange{toU32(0), toU32(10), toU32(3), BinaryOp::Lt}, body));
    CHECK_NOTHROW(For(ForRange{toI32(10), toI32(0), toI32(-3), BinaryOp::Ge}, body));
    out.set(vec4(1.0));
  });
}

TEST_CASE("an array layout larger than memory is rejected") {
  // 2^30 * 2^30 * 16 bytes is 2^64: it would wrap to 0 on a 64 bit host
  const TypePtr huge = arrayOf(arrayOf(vec4f(), 1 << 30), 1 << 30);
  CHECK_THROWS_AS(storageLayout(huge), ShaderError);
  CHECK_THROWS_AS(structLayout(makeStruct("Huge", {{"a", huge}})), ShaderError);
  // two arrays that fit alone but not together
  const TypePtr half = arrayOf(arrayOf(vec4f(), 1 << 30), 1 << 29);
  CHECK_THROWS_AS(storageLayout(makeStruct("Halves", {{"a", half}, {"b", half}})), ShaderError);
  CHECK(storageLayout(arrayOf(vec4f(), 4)).size == 64);
}

TEST_CASE("two different varyings may not share a name") {
  const Varying a = varying("tone", f32());
  const Varying b = varying("tone", f32());
  const Output out = output(0);
  CHECK_THROWS_WITH_AS(analyzeProgram(program("t", "T", [&] {
                         a.set(1.0);
                         b.set(2.0);
                         Position(vec4(0.0));
                       }, [&] { out.set(vec4(a.value())); })),
                       doctest::Contains("Two different varyings are named \"tone\""), ShaderError);
}

TEST_CASE("helper functions built outside a stage may hold stage operations") {
  // built before any program: the stage is checked where each one is called
  const Function sync = fn("sync", {}, voidType(), [](const Args&) { workgroupBarrier(); });
  const Function total = fn("total", {{"x", f32()}}, f32(), [](const Args& a) { Return(subgroupAdd(a["x"])); });
  const Function cut = fn("cut", {}, voidType(), [](const Args&) { Discard(); });
  const Output out = output(0);
  CHECK_NOTHROW(analyzeProgram(computeProgram("c", "C", {64, 1, 1}, [&] {
    sync();
    const Val s = Let(total(1.0), "s");
  })));
  CHECK_NOTHROW(analyzeProgram(program("f", "F", [] { Position(vec4(0.0)); }, [&] {
    cut();
    out.set(vec4(total(1.0)));
  })));
  CHECK_THROWS_WITH_AS(analyzeProgram(program("v", "V", [&] {
                         sync();
                         Position(vec4(0.0));
                       }, [&] { out.set(vec4(1.0)); })),
                       doctest::Contains("workgroupBarrier() is only valid in the compute stage"), ShaderError);
  CHECK_THROWS_AS(analyzeProgram(computeProgram("d", "D", {64, 1, 1}, [&] { cut(); })), ShaderError);
  CHECK_THROWS_AS(analyzeProgram(program("w", "W", [&] {
                    Position(vec4(total(1.0)));
                  }, [&] { out.set(vec4(1.0)); })),
                  ShaderError);
  // inside a stage the check stays immediate
  CHECK_THROWS_AS(program("i", "I", [] {
                    workgroupBarrier();
                    Position(vec4(0.0));
                  }, [] {}),
                  ShaderError);
}

TEST_CASE("UniformWriter writes integer vectors and uploads only its layout") {
  const auto layout = structLayout(makeStruct("Ints", {{"counts", vec4u()}, {"offsets", vec2i()}}));
  std::vector<std::byte> big(layout->size + 64);
  UniformWriter w(layout, big);
  // bytes() is the layout's part of a larger external buffer
  CHECK(w.bytes().size() == layout->size);
  const std::array<std::uint32_t, 4> counts{1, 2, 3, 4294967295u};
  w.setUVec("counts", counts);
  const std::array<std::int32_t, 2> offsets{-1, 7};
  w.setIVec("offsets", offsets);
  std::array<std::uint32_t, 4> c{};
  std::memcpy(c.data(), w.bytes().data(), 16);
  CHECK(c == counts);
  std::array<std::int32_t, 2> o{};
  std::memcpy(o.data(), w.bytes().data() + layout->find("offsets")->offset, 8);
  CHECK(o == offsets);
  CHECK_THROWS_AS(w.setIVec("counts", offsets), ShaderError);
  const std::array<std::uint32_t, 5> tooMany{};
  CHECK_THROWS_AS(w.setUVec("counts", tooMany), ShaderError);
}

TEST_CASE("names that are GLSL macros are renamed") {
  const Val es = constant("GL_ES", 2.0);
  const UniformBlock block = uniformBlock("macros", {{"GL_FRAGMENT_PRECISION_HIGH", f32()}});
  const Output out = output(0);
  const ProgramAST ast = program("g", "G", [] { Position(vec4(0.0)); },
                                 [&] { out.set(vec4(es * block["GL_FRAGMENT_PRECISION_HIGH"])); });
  const CompiledGLSL g = compileGLSL(analyzeProgram(ast));
  // GLSL ES predefines GL_ES and GL_FRAGMENT_PRECISION_HIGH; a GL_ name would be replaced by the macro
  CHECK(has(g.fragment, "xGL_ES"));
  CHECK(has(g.fragment, "xGL_FRAGMENT_PRECISION_HIGH"));
  CHECK_FALSE(has(g.fragment, " GL_ES"));
}

TEST_CASE("select is a constant expression with a known value") {
  // WGSL evaluates select at shader creation (Tint, Chrome 155: a const select compiles)
  const Val chosen = constant("CHOSEN", select(true, toI32(1), toI32(2)));
  const Val picks = constant("PICKS", select(bvec2(true, false), ivec2(3, 4), ivec2(1, 2)));
  const Output out = output(0);
  program("s", "S", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    const Val w = Let(vec4(1.0), "w");
    const Val x = Let(toF32(fragCoord().x()), "x");
    // Tint: "index 5 out of bounds [0..1]"
    CHECK_THROWS_WITH_AS(v[select(true, toI32(5), toI32(0))], doctest::Contains("Index 5 is out of range"), ShaderError);
    // Tint: "clamp called with low (2.0) greater than high (1.0)"
    CHECK_THROWS_AS(clamp(x, select(true, 2.0, 0.0), 1.0), ShaderError);
    // per component: (3, 4) where true, (1, 2) where false gives (3, 2); w[2] is in range
    CHECK(w[picks.y()].type() == f32());
    CHECK_THROWS_AS(w[picks.y() + toI32(2)], ShaderError);
    CHECK(v[chosen].type() == f32());
    out.set(vec4(1.0));
  });
}

TEST_CASE("an array constructor checks its argument count before anything else") {
  // 2^30 elements: listing their types first would allocate gigabytes before the error
  const auto start = std::chrono::steady_clock::now();
  CHECK_THROWS_WITH_AS(construct(arrayOf(f32(), 1 << 30), {1.0}),
                       doctest::Contains("array<f32, 1073741824>: expected 1073741824 values, got 1"), ShaderError);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
}

TEST_CASE("a constant from a reused value is checked in linear time") {
  Val v = toF32(1.0);
  // 40 doublings: 41 nodes, 2^40 leaves when written out as a tree
  for (int i = 0; i < 40; ++i) v = v + v;
  const auto start = std::chrono::steady_clock::now();
  const Val big = constant("BIG_REUSE", v);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  (void)big;
}

TEST_CASE("a constant built from a reused value prints each reused part once") {
  Val zero = toI32(0);
  Val big = toF32(1.0);
  for (int i = 0; i < 40; ++i) {
    zero = zero + zero;
    big = big + big * 0.0;
  }
  const Val offsetX = constant("SHARED_OFFSET", zero);
  const Val grown = constant("SHARED_GROWN", big);
  // a runtime constant (INFINITY is a WGSL function) gets lets inside its function
  const Val runtime = constant("SHARED_RUNTIME", big * 0.0 + consts::infinity() * 0.0 + big);
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Output out = output(0);
  const ProgramAST ast = program("c", "C", [] { Position(vec4(0.0)); }, [&] {
    // a texel offset must be a constant expression: the helper constants keep it one
    out.set(albedo.sampleOffset(vec2(0.5), ivec2(offsetX, 0)) * grown * runtime);
  });
  const auto start = std::chrono::steady_clock::now();
  const AnalyzedProgram a = analyzeProgram(ast);
  const std::string w = compileWGSL(a).code;
  const CompiledGLSL g = compileGLSL(a);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  CHECK(w.size() < 20000);
  CHECK(g.fragment.size() < 20000);
  CHECK(has(w, "const moga_shared_"));
  CHECK(has(w, "  let moga_shared_"));
  CHECK(has(g.fragment, "const int moga_shared_"));
}

TEST_CASE("a constant smoothstep with reversed edges is a GLSL constant expression") {
  const Output out = output(0);
  const Val c = constant("fade", smoothstep(1.0, 0.0, 0.25));
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
  const std::string glsl = compileGLSL(analyzeProgram(ast)).fragment;
  // GLSL ES 3.00 4.3.3: a const initializer may call builtins only, never a user function
  const std::size_t at = glsl.find("const float fade = ");
  REQUIRE(at != std::string::npos);
  CHECK_FALSE(has(glsl.substr(0, glsl.find('\n', at)), "moga_smoothstep"));
}

TEST_CASE("a bitcast inside a select is rejected in a constant") {
  const Val inf = uintBitsToFloat(toU32(0x7F800000u));
  CHECK_THROWS_WITH_AS(constant("pick", select(true, inf, toF32(0.0))),
                       doctest::Contains("a bitcast must be the whole value"), ShaderError);
}

TEST_CASE("a constant named roundEven is renamed in GLSL") {
  const Output out = output(0);
  const Val c = constant("roundEven", 1.0);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
  const std::string glsl = compileGLSL(analyzeProgram(ast)).fragment;
  // roundEven is a GLSL ES 3.00 builtin (8.3), so a global of that name would hide it
  CHECK_FALSE(has(glsl, "const float roundEven "));
}

TEST_CASE("every GLSL ES 3.00 texture function name is renamed for a constant") {
  // GLSL ES 3.00 8.8 lists these; a global of the same name hides the builtin
  for (const char* name : {"texture", "textureProjOffset", "textureProjLod", "textureProjLodOffset",
                           "textureGradOffset", "textureProjGrad", "textureProjGradOffset"}) {
    const Output out = output(0);
    const Val c = constant(name, 1.0);
    const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
    const std::string glsl = compileGLSL(analyzeProgram(ast)).fragment;
    CHECK_FALSE(has(glsl, std::format("const float {} ", name)));
  }
}

TEST_CASE("a constant smoothstep with edges the DSL cannot evaluate is a GLSL constant expression") {
  const Output out = output(0);
  const Val c = constant("s", smoothstep(sin(toF32(0.0)), 1.0, 0.5));
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
  const std::string glsl = compileGLSL(analyzeProgram(ast)).fragment;
  const std::size_t at = glsl.find("const float s = ");
  REQUIRE(at != std::string::npos);
  // GLSL ES 3.00 4.3.3: only builtins in a const initializer
  CHECK_FALSE(has(glsl.substr(0, glsl.find('\n', at)), "moga_smoothstep"));
}

TEST_CASE("nested constant smoothsteps print linear GLSL") {
  const Output out = output(0);
  Val v = toVal(0.5);
  for (int i = 0; i < 40; ++i) v = smoothstep(1.0, 0.0, v);
  const Val c = constant("deep", v);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
  const std::string glsl = compileGLSL(analyzeProgram(ast)).fragment;
  // 40 levels written out three times each would be 3^40 copies; one helper const each stays small
  CHECK(glsl.size() < 20000);
  CHECK(has(glsl, "const float deep = "));
}

TEST_CASE("constant member and index expressions are constants") {
  // WGSL 8.x and GLSL ES 3.00 4.3.3: a component of a constant vector, an element of a constant array
  // and a field of a constant struct are constant expressions
  const Val k = constant("k", ivec2(1, 2)[0]);
  const Val e = constant("e", construct(arrayOf(f32(), 3), {1.5, 2.5, 3.5})[2]);
  const TypePtr pair = makeStruct("KPair", {{"a", f32()}, {"b", f32()}});
  const Val m = constant("m", construct(pair, {4.0, 7.0}).member("b"));
  // the values are known: each one as a low bound above 1.0 is a reversed clamp
  CHECK_THROWS_WITH_AS(clamp(toF32(0.0), e, 1.0), doctest::Contains("low bound 3.5"), ShaderError);
  CHECK_THROWS_WITH_AS(clamp(toF32(0.0), m, 1.0), doctest::Contains("low bound 7"), ShaderError);
  CHECK_THROWS_WITH_AS(clamp(toI32(0), k + 1, 1), doctest::Contains("low bound 2"), ShaderError);
}

TEST_CASE("an index known through another index is bounds checked") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // ivec2(5, 0)[0] is 5, past the last component (1) of a vec2
    CHECK_THROWS_AS((void)v[ivec2(5, 0)[0]], ShaderError);
    out.set(vec4(v[ivec2(1, 0)[0]]));
  });
}

TEST_CASE("mod rejects a known zero divisor of a known dividend") {
  // 1 - 0 * floor(1 / 0) is NaN: WGSL rejects a constant expression that is not finite
  CHECK_THROWS_WITH_AS(mod(toF32(1.0), toF32(0.0)), doctest::Contains("mod: the divisor is zero"), ShaderError);
  CHECK_THROWS_AS(mod(vec2(1.0, 2.0), vec2(1.0, 0.0)), ShaderError);
  CHECK_NOTHROW(mod(toF32(1.0), toF32(3.0)));
}

TEST_CASE("smoothstep sees equal edges through fract") {
  // fract(0.0) is 0.0, the same as the high edge
  CHECK_THROWS_WITH_AS(smoothstep(fract(toF32(0.0)), 0.0, 0.5), doctest::Contains("both edges are 0"), ShaderError);
  CHECK_NOTHROW(smoothstep(fract(toF32(1.25)), 1.0, 0.5));
}

TEST_CASE("nested mods print linear WGSL") {
  const Output out = output(0);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    Val x = Let(0.5, "x");
    Val v = x;
    for (int i = 0; i < 40; ++i) v = mod(v * 3.0, 2.0);
    out.set(vec4(v));
  });
  const std::string wgsl = compileWGSL(analyzeProgram(ast)).code;
  // doubling at each of 40 levels would be 2^40 copies of x
  CHECK(wgsl.size() < 20000);
  // a single node operand keeps the short form
  const ProgramAST small = program("q", "Q", [] { Position(vec4(0.0)); }, [&] {
    const Val y = Let(0.5, "y");
    out.set(vec4(mod(y, 2.0)));
  });
  CHECK(has(compileWGSL(analyzeProgram(small)).code, " - 2.0f * floor("));
}

TEST_CASE("nested constant mods print linear WGSL constants") {
  const Output out = output(0);
  Val v = toVal(0.5);
  for (int i = 0; i < 40; ++i) v = mod(v * 3.0, 2.0);
  const Val c = constant("deepMod", v);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
  const std::string wgsl = compileWGSL(analyzeProgram(ast)).code;
  CHECK(wgsl.size() < 20000);
  const std::size_t at = wgsl.find("const deepMod: f32 = ");
  REQUIRE(at != std::string::npos);
  // a const initializer cannot call the moga_mod helper
  CHECK_FALSE(has(wgsl.substr(0, wgsl.find('\n', at)), "moga_mod"));
}

TEST_CASE("a local used after the block that declares it is rejected") {
  const Output out = output(0);
  // WGSL and GLSL ES 3.00: a local is visible only in its enclosing block
  std::optional<Val> kept;
  const ProgramAST escaped = program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(0.5, "x");
    If(x > 0.0, [&] { kept = Var(1.0, "inner"); });
    out.set(vec4(*kept));
  });
  CHECK_THROWS_WITH_AS(analyzeProgram(escaped), doctest::Contains("Local inner is used outside the block that declares it"),
                       ShaderError);

  // with debug info, the error starts at the statement that uses it
  std::uint32_t line = 0;
  const BuildResult located = buildShader("q", "Q", [] { Position(vec4(0.0)); }, [&] {
    If(true, [&] { kept = Var(1.0, "inner"); });
    line = __LINE__ + 1;
    out.set(vec4(*kept));
  }, {.sourceLocations = true});
  REQUIRE(located.diagnostics.size() == 1);
  CHECK(located.diagnostics[0].code == ErrorCode::Scope);
  REQUIRE(located.diagnostics[0].where);
  CHECK(located.diagnostics[0].where->line == line);

  // the same local inside its block, and a loop variable in its loop, are fine
  const ProgramAST fine = program("r", "R", [] { Position(vec4(0.0)); }, [&] {
    const Val acc = Var(0.0, "acc");
    If(true, [&] {
      const Val inner = Var(1.0, "inner");
      acc.set(acc + inner);
    });
    For(0, 4, [&](const Val& i) { acc.set(acc + toF32(i)); });
    out.set(vec4(acc));
  });
  CHECK_NOTHROW(analyzeProgram(fine));
}

TEST_CASE("a constant bool conversion is known") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // bool(1) is true, so the select picks 5: past the last component (1) of a vec2
    CHECK_THROWS_AS((void)v[select(toBool(toI32(1)), toI32(5), toI32(0))], ShaderError);
    out.set(vec4(v[select(toBool(toI32(0)), toI32(5), toI32(1))]));
  });
}

TEST_CASE("an array field of a constant struct is known") {
  const Output out = output(0);
  const TypePtr holder = makeStruct("Holder", {{"items", arrayOf(i32(), 2)}});
  const Val h = constant("h", construct(holder, {construct(arrayOf(i32(), 2), {1, 7})}));
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // items[1] is 7, past the last component (1) of a vec2
    CHECK_THROWS_AS((void)v[h.member("items")[1]], ShaderError);
    out.set(vec4(v[h.member("items")[0]]));
  });
}

TEST_CASE("a constant integer dot is known") {
  // (1, 2) . (2, -1) = 2 - 2 = 0: WGSL rejects an integer division by a constant zero
  CHECK_THROWS_AS(toI32(4) / dot(ivec2(1, 2), ivec2(2, -1)), ShaderError);
  CHECK_NOTHROW(toI32(4) / dot(ivec2(1, 2), ivec2(2, 1)));
  // 65536 * 65536 wraps to 0 in u32
  CHECK_THROWS_AS(toU32(4u) / dot(uvec2(65536u, 0u), uvec2(65536u, 0u)), ShaderError);
}

TEST_CASE("a constant cannot take the name a local prints as") {
  const Output out = output(0);
  const Val c = constant("x_1000000", 2.0);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Var(1.0, "x");
    out.set(vec4(x + c));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  // a local named x prints as x_<id>; the constant must keep a name no local can take
  for (const std::string& code : {compileGLSL(a).fragment, compileWGSL(a).code}) {
    CHECK_FALSE(has(code, "x_1000000 "));
    CHECK_FALSE(has(code, "x_1000000:"));
  }
}

TEST_CASE("constants are renamed away from the WGSL f16 matrix aliases") {
  const Output out = output(0);
  for (const char* name : {"mat2x2h", "mat3x4h", "mat4x4h"}) {
    const Val c = constant(name, 1.0);
    const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(c)); });
    CHECK_FALSE(has(compileWGSL(analyzeProgram(ast)).code, std::format("const {}:", name)));
  }
}

TEST_CASE("a constant float to integer conversion clamps as WGSL does") {
  // WGSL: u32(-1.0) is 0u and i32(3e9) is 2147483520, the largest f32 below 2^31
  CHECK_THROWS_AS(toU32(4u) / toU32(toF32(-1.0)), ShaderError);
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    CHECK_THROWS_AS((void)v[toI32(toF32(3e9))], ShaderError);
    CHECK_THROWS_AS((void)v[toI32(toF32(-3e9))], ShaderError);
    out.set(vec4(v[toI32(toF32(1.9))]));
  });
}

TEST_CASE("a constant matrix element is known") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // mat4(5.0) has 5 on its diagonal and 0 elsewhere
    CHECK_THROWS_AS((void)v[toI32(mat4(5.0)[0][0])], ShaderError);
    CHECK_NOTHROW((void)v[toI32(mat4(5.0)[0][1])]);
    // from columns, and from every component in column order: m[1][0] is 3
    CHECK_THROWS_AS((void)v[toI32(mat2(vec2(0.0, 1.0), vec2(3.0, 0.0))[1][0])], ShaderError);
    CHECK_THROWS_AS((void)v[toI32(mat2(0.0, 1.0, 3.0, 0.0)[1][0])], ShaderError);
    out.set(vec4(v[toI32(mat2(0.0, 1.0, 3.0, 0.0)[0][1])]));
  });
}

TEST_CASE("a local never takes a name that a module declaration can have") {
  const Output out = output(0);
  const Function foo = fn("foo_1", {}, f32(), [](const Args&) { Return(1.0); });
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    // f_foo plus an id has the shape of function foo_1 (f_foo_1); o plus an id the shape of output 0 (o_0)
    const Val f = Var(2.0, "f_foo");
    const Val o = Var(3.0, "o");
    out.set(vec4(f + o + foo()));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  for (const std::string& code : {compileGLSL(a).fragment, compileWGSL(a).code}) {
    CHECK(has(code, "l_f_foo_"));
    CHECK(has(code, "l_o_"));
    CHECK(has(code, "f_foo_1("));
  }
}

TEST_CASE("constant any and all are known") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // any((false, true)) is true and all((false, true)) is false
    CHECK_THROWS_AS((void)v[select(any(bvec2(false, true)), toI32(5), toI32(0))], ShaderError);
    CHECK_THROWS_AS((void)v[select(all(bvec2(false, true)), toI32(0), toI32(5))], ShaderError);
    out.set(vec4(v[select(all(bvec2(false, true)), toI32(5), toI32(0))]));
  });
}

TEST_CASE("a constant float dot is known when its rounding cannot matter") {
  // (0, 0) . (0, 0) = 0, so an integer made from it is a zero divisor
  CHECK_THROWS_AS(toI32(4) / toI32(dot(vec2(0.0, 0.0), vec2(0.0, 0.0))), ShaderError);
  CHECK_THROWS_AS(toI32(4) / toI32(dot(vec3(1.0, 2.0, 3.0), vec3(3.0, 0.0, -1.0))), ShaderError);
  // 1e8 + 1 - 1e8 depends on the order of the sum in f32 (1 or 0), so the divisor can be zero
  CHECK_THROWS_AS(toI32(4) / toI32(dot(vec3(1e8, 1.0, -1e8), vec3(1.0, 1.0, 1.0))), ShaderError);
}

TEST_CASE("a large local array is zeroed by a loop in GLSL") {
  const Output out = output(0);
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val big = Var(arrayOf(vec4f(), 1 << 30), "big");
    const Val small = Var(arrayOf(f32(), 4), "small");
    out.set(big[0] + vec4(small[1]));
  });
  const auto start = std::chrono::steady_clock::now();
  const std::string glsl = compileGLSL(analyzeProgram(ast)).fragment;
  // one constructor argument per element would be a billion strings
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
  CHECK(has(glsl, " < 1073741824; ++moga_i_"));
  CHECK(has(glsl, "] = vec4(0);"));
  CHECK(has(glsl, "float[4](0.0, 0.0, 0.0, 0.0)"));
}

TEST_CASE("an element of a matrix built from a matrix is known") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // mat3 from mat4(5.0) keeps the top left part: 5 on the diagonal
    CHECK_THROWS_AS((void)v[toI32(construct(mat3f(), {mat4(5.0)})[2][2])], ShaderError);
    CHECK_THROWS_AS((void)v[toI32(construct(mat2h(), {mat2(5.0)})[1][1])], ShaderError);
    out.set(vec4(v[toI32(construct(mat3f(), {mat4(5.0)})[2][0])]));
  });
}

TEST_CASE("a value through an f16 matrix is stored in f16") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 2049 is not an f16 (the spacing is 2 from 2048 to 4096): the f16 matrix holds 2048 or 2050,
    // so 2049 - that is 1 or -1, and -1 is out of range
    const Val m = construct(mat2f(), {construct(mat2h(), {mat2(2049.0)})});
    CHECK_THROWS_AS((void)v[toI32(toF32(2049.0) - m[0][0])], ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("an integer to f32 conversion may give either neighbor") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 2^24 + 1 is not an f32: f32(16777217u) is 16777216 or 16777218, so the index is 0 or 2
    CHECK_THROWS_AS((void)v[toI32(toF32(toU32(16777217u)) - 16777216.0)], ShaderError);
    // 2^24 + 2 is an f32, so the index is exactly 2: one value, reported as such
    CHECK_THROWS_WITH_AS((void)v[toI32(toF32(toU32(16777218u)) - 16777216.0)], doctest::Contains("Index 2 is out of range"),
                         ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("a float dot may round either way") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 1 - 2^-25 is between the f32 values 1 - 2^-24 and 1: i32 of it is 0 or 1, and
    // 2 - that index is 2 or 1; 2 is out of range
    const Val d = dot(vec2(1.0, -std::ldexp(1.0, -25)), vec2(1.0, 1.0));
    CHECK_THROWS_AS((void)v[toI32(2) - toI32(d)], ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("an exact zero f16 dot is known") {
  // 0 * 0 + 0 * 0 is 0 in every order and rounding
  CHECK_THROWS_AS(toI32(4) / toI32(dot(hvec2(0.0, 0.0), hvec2(0.0, 0.0))), ShaderError);
}

TEST_CASE("a subnormal float result may be flushed to zero") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 2^-126 * 0.5 is the subnormal 2^-127, which WGSL may flush to 0: bool of it is true or false
    const Val tiny = dot(vec2(std::ldexp(1.0, -126), 0.0), vec2(0.5, 0.0));
    CHECK_THROWS_AS((void)v[select(toBool(tiny), toI32(1), toI32(5))], ShaderError);
    out.set(vec4(1.0));
  });
}

namespace {
/** 2^24 + 1 is not an f32: f32(16777217u) is 16777216 or 16777218, so this is 0 or 2. */
Val zeroOrTwo() { return toF32(toU32(16777217u)) - 16777216.0; }
} // namespace

TEST_CASE("two uses of one uncertain constant hold the same value") {
  const Output out = output(0);
  const Val c = constant("ZERO_OR_TWO", zeroOrTwo());
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(0.5, "x");
    // the bounds are the same value, 0 or 2, so they are equal either way
    CHECK_NOTHROW(clamp(x, c, c));
    CHECK_NOTHROW(clamp(vec2(x), c, c));
    // separate computations can differ: 0 and 2, or 2 and 0
    CHECK_THROWS_AS(clamp(x, zeroOrTwo(), zeroOrTwo()), ShaderError);
    CHECK_THROWS_AS(smoothstep(c, c, x), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("normalize rejects a vector that can be zero") {
  // (0, -0) when the conversion gives 16777216
  CHECK_THROWS_WITH_AS(normalize(vec2(zeroOrTwo(), -zeroOrTwo())), doctest::Contains("can be zero"), ShaderError);
  CHECK_NOTHROW(normalize(vec2(zeroOrTwo(), 1.0)));
}

TEST_CASE("a loop end that can be the largest i32 is a wrap") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    // the end is 2147483646 or 2147483647; with <= the counter passes 2147483647 and wraps
    const Val end = toI32(2147483646) + toI32(zeroOrTwo() / 2.0);
    CHECK_THROWS_AS(For(ForRange{toI32(2147483646), end, toI32(1), BinaryOp::Le}, [](const Val&) {}), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("a left shift is checked for every possible amount") {
  // the amount is 30 or 31; 2 << 31 does not fit in u32
  CHECK_THROWS_AS(toU32(2u) << (toU32(30u) + toU32(zeroOrTwo() / 2.0)), ShaderError);
  CHECK_NOTHROW(toU32(1u) << (toU32(30u) + toU32(zeroOrTwo() / 2.0)));
}

TEST_CASE("an integer range across the wrap covers both ends") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 0 or 2, plus 0xffffffff, wraps to 0xffffffff or 1; 0xffffffff is out of range
    CHECK_THROWS_AS((void)v[toU32(zeroOrTwo()) + toU32(0xffffffffu)], ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("one expression node used twice may round differently in each copy") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(0.5, "x");
    // a short value is printed at each use, and each copy may be 0 or 2
    const Val v = zeroOrTwo();
    CHECK_THROWS_AS(clamp(x, v, v), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("an f32 value prints as its exact decimal") {
  const Output out = output(0);
  const float largest = std::numeric_limits<float>::max();
  const float tenth = 0.1f;
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    out.set(vec4(toF32(static_cast<double>(largest)), toF32(static_cast<double>(tenth)), 0.0, 1.0));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  // the largest f32 is (2 - 2^-23) * 2^127; the shortest double decimal, 3.4028234663852886e+38, is above it
  // 0.1f is 13421773 * 2^-27; its shortest double decimal 0.10000000149011612 is not equal to it
  for (const std::string& code : {compileWGSL(a).code, compileGLSL(a).fragment}) {
    CHECK(has(code, "340282346638528859811704183484516925440.0"));
    CHECK(has(code, "0.100000001490116119384765625"));
  }
}

TEST_CASE("an exact subnormal literal is never flushed") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 2^-148 is an f32 subnormal written as a literal: bool of it is true, so the index is 1
    CHECK_NOTHROW((void)v[select(toBool(toF32(std::ldexp(1.0, -148))), toI32(1), toI32(5))]);
    out.set(vec4(1.0));
  });
}

TEST_CASE("mix is checked for every combination of possible values") {
  // a is 0 or about 1.2e38 and t is -2 or 3: a * (1 - t) with a = 1.2e38 and t = -2 is 3.6e38, past the largest f32
  const Val a = zeroOrTwo() * 6.0e37;
  const Val t = zeroOrTwo() * 2.5 - 2.0;
  CHECK_THROWS_AS(mix(a, toF32(0.0), t), ShaderError);
  CHECK_NOTHROW(mix(zeroOrTwo(), toF32(0.0), t));
}

TEST_CASE("a float dot overflows when some order of its sum does") {
  // 2e38 - 2e38 + 2e38 is finite in that order; 2e38 + 2e38 first is 4e38, past the largest f32
  CHECK_THROWS_AS(dot(vec3(2.0e38, -2.0e38, 2.0e38), vec3(1.0)), ShaderError);
  CHECK_NOTHROW(dot(vec3(1.0e38, -2.0e38, 1.0e38), vec3(1.0)));
}

TEST_CASE("a float division may be off by 2.5 ULP") {
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // 5 / 5 may be just below 1, which truncates to 0: the index 2 - i32(5 / 5) - i32(5 / 5) can be 2
    CHECK_THROWS_AS((void)v[toI32(2) - toI32(toF32(5.0) / 5.0) - toI32(toF32(5.0) / 5.0)], ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("a subnormal operand may be read as zero") {
  const Output out = output(0);
  const double tiny = std::ldexp(1.0, -140);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // sign(2^-140) is 1, or 0 when the subnormal input is flushed: the index is 1 or 2
    CHECK_THROWS_AS((void)v[toI32(2) - toI32(sign(toF32(tiny)))], ShaderError);
    // max(2^-140, 2^-141) is a subnormal, or 0 when both inputs are flushed
    CHECK_THROWS_AS((void)v[select(toBool(max(toF32(tiny), toF32(tiny / 2.0))), toI32(1), toI32(5))], ShaderError);
    // -2^-140 may be flushed to zero
    CHECK_THROWS_AS((void)v[select(toBool(-toF32(tiny)), toI32(1), toI32(5))], ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("a matrix conversion keeps exact values and reports overflow") {
  // 70000 is past the largest f16 (65504)
  CHECK_THROWS_AS(construct(mat2h(), {mat2(70000.0)}), ShaderError);
  const Output out = output(0);
  program("p", "P", [] { Position(vec4(0.0)); }, [&] {
    const Val v = Let(vec2(1.0, 2.0), "v");
    // a same type conversion copies the subnormal 2^-140, never zero, so bool of it is true
    const Val m = construct(mat2f(), {mat2(std::ldexp(1.0, -140))});
    CHECK_NOTHROW((void)v[select(toBool(m[0][0]), toI32(1), toI32(5))]);
    out.set(vec4(1.0));
  });
}
