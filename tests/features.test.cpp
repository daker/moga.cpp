#include <doctest/doctest.h>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

using namespace moga;

namespace {

bool has(const std::string& haystack, std::string_view needle) {
  return haystack.find(needle) != std::string::npos;
}

struct Both {
  std::string glsl;
  std::string wgsl;
};

/** Compiles a fragment body that writes `out` on both backends. */
template <class F> Both fragment(F&& body) {
  const Output out = output(0);
  const ProgramAST ast = program("t", "T", [] { Position(vec4(0.0)); }, [&] { body(out); });
  const AnalyzedProgram a = analyzeProgram(ast);
  return {compileGLSL(a).fragment, compileWGSL(a).code};
}

} // namespace

TEST_CASE("missing constructors build the expected types") {
  fragment([](const Output& out) {
    CHECK(ivec4(1).type() == vec4i());
    CHECK(uvec3(1).type() == vec3u());
    CHECK(bvec2(true, false).type() == vec2b());
    CHECK(mat2(1.0).type() == mat2f());
    CHECK(color(1.0, 0.5, 0.0).type() == vec3f());
    CHECK(toBool(toF32(2.0)).type() == boolType());
    CHECK(toBool(0).type() == boolType());
    out.set(vec4(1.0));
  });
}

TEST_CASE("convert changes size and kind") {
  const Both b = fragment([](const Output& out) {
    const Val v4 = Let(vec4(1.0, 2.0, 3.0, 4.0), "v4");
    CHECK(toVec2(v4).type() == vec2f());
    CHECK(toIVec3(v4).type() == vec3i());
    CHECK(toVec4(vec2(1.0)).type() == vec4f());
    const Val p = Let(toVec4(vec2(1.0)), "p");
    out.set(p);
  });
  // a smaller vector gets zero components at the end
  CHECK(has(b.glsl, "vec4(vec2(1.0), 0.0, 0.0)"));
  CHECK(has(b.wgsl, "vec4f(vec2f(1.0f), 0.0f, 0.0f)"));
}

TEST_CASE("compound assignment and increment") {
  const Both b = fragment([](const Output& out) {
    const Val i = Var(toI32(7), "i");
    i.remAssign(4);
    i.shlAssign(1);
    i.bitXorAssign(3);
    i.increment();
    const Val f = Var(8.0, "f");
    f.divAssign(2.0);
    out.set(vec4(f + toF32(i)));
  });
  // native compound assignment evaluates the target once
  CHECK(has(b.glsl, " %= 4;"));
  CHECK(has(b.glsl, " <<= 1;"));
  CHECK(has(b.glsl, " ^= 3;"));
  CHECK(has(b.glsl, " += 1;"));
  CHECK(has(b.glsl, " /= 2.0;"));
  CHECK(has(b.wgsl, " %= 4i;"));
  // WGSL shift amounts are unsigned
  CHECK(has(b.wgsl, " <<= u32(1i);"));
}

TEST_CASE("switch never falls through and WGSL always has a default") {
  const Both b = fragment([](const Output& out) {
    const Val c = Var(vec3(0.0), "c");
    Switch(toI32(fragCoord().x()))
        .Case({0, 1}, [&] { c.set(vec3(1.0, 0.0, 0.0)); })
        .Case(2, [&] { c.set(vec3(0.0, 1.0, 0.0)); });
    out.set(vec4(c, 1.0));
  });
  CHECK(has(b.glsl, "case 0:\n"));
  CHECK(has(b.glsl, "case 1: {"));
  CHECK(has(b.glsl, "break;"));
  CHECK(has(b.wgsl, "case 0i, 1i: {"));
  CHECK(has(b.wgsl, "default: {"));
}

TEST_CASE("switch values must be literals of an integer selector") {
  fragment([](const Output& out) {
    CHECK_THROWS_AS(Switch(1.5), ShaderError);
    CHECK_THROWS_AS(Switch(toI32(1)).Case(toI32(2), [] {}), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("loop forms") {
  const Both b = fragment([](const Output& out) {
    const Val sum = Var(0.0, "sum");
    For({.start = 10, .end = 0, .step = -2, .condition = BinaryOp::Gt},
        [&](const Val& i) { sum.addAssign(toF32(i)); });
    ForGrid(2, 3, [&](const Val& i, const Val& j) { sum.addAssign(toF32(i * j)); });
    While(sum < 100.0, [&] { sum.mulAssign(2.0); });
    out.set(vec4(sum));
  });
  CHECK(has(b.glsl, " = 10; i_"));
  CHECK(has(b.glsl, " > 0; i_"));
  CHECK(has(b.glsl, " += (-2))"));
  CHECK(has(b.glsl, "while ((sum_"));
  CHECK(has(b.wgsl, " = i_"));
  CHECK(has(b.wgsl, "while ((sum_"));
  CHECK_THROWS_AS(fragment([](const Output&) {
                    For({.start = toU32(3), .end = 0, .step = -1, .condition = BinaryOp::Gt},
                        [](const Val&) {});
                  }),
                  ShaderError);
}

TEST_CASE("math functions use each backend's name") {
  const Both b = fragment([](const Output& out) {
    const Val x = Let(fragCoord().x(), "x");
    const Val h = Let(sinh(x) + atanh(x) + radians(x) + degrees(x), "h");
    const Val r = Let(refract(vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), 0.75), "r");
    const Val p = Let(packHalf2x16(vec2(x, h)), "p");
    const Val u = Let(unpackUnorm2x16(p), "u");
    out.set(vec4(r * oneMinus(x) * pow3(h), u.x()));
  });
  CHECK(has(b.glsl, "sinh(x_"));
  CHECK(has(b.glsl, "atanh(x_"));
  CHECK(has(b.glsl, "refract("));
  CHECK(has(b.glsl, "packHalf2x16("));
  CHECK(has(b.glsl, "unpackUnorm2x16(p_"));
  CHECK(has(b.wgsl, "pack2x16float("));
  CHECK(has(b.wgsl, "unpack2x16unorm(p_"));
  CHECK(has(b.wgsl, "(1.0f - x_"));
}

TEST_CASE("cube root keeps the sign") {
  const Both b = fragment([](const Output& out) { out.set(vec4(cbrt(fragCoord().x()))); });
  CHECK(has(b.glsl, "(sign(gl_FragCoord.x) * pow(abs(gl_FragCoord.x), 0.3333333333333333))"));
}

TEST_CASE("inverse is native in GLSL and a helper in WGSL") {
  const UniformBlock ub = uniformBlock("xf", {{"m", mat4f()}, {"n", mat3f()}});
  const Both b = fragment([&](const Output& out) {
    const Val a = Let(inverse(ub["m"]) * vec4(1.0), "a");
    const Val c = Let(inverse(ub["n"]) * vec3(1.0), "c");
    out.set(a + vec4(c, 0.0));
  });
  CHECK(has(b.glsl, "inverse(ub_xf.m)"));
  CHECK(has(b.wgsl, "fn moga_inverse_mat4x4f(m: mat4x4f) -> mat4x4f {"));
  CHECK(has(b.wgsl, "fn moga_inverse_mat3x3f(m: mat3x3f) -> mat3x3f {"));
  CHECK(has(b.wgsl, "moga_inverse_mat4x4f(ub_xf.m)"));
  CHECK_FALSE(has(b.wgsl, "moga_inverse_mat2x2f"));
  CHECK_THROWS_AS(fragment([](const Output&) { inverse(vec3(1.0)); }), ShaderError);
}

TEST_CASE("single uniforms are packed per frequency, sorted by name") {
  const Val tint = uniform("tint", vec4f());
  const Val gain = uniform("gain", f32());
  const Val time = uniform("time", f32(), ResourceFrequency::Frame);
  const Output out = output(0);
  const ProgramAST ast = program(
      "u", "U", [] { Position(vec4(0.0)); }, [&] { out.set(tint * gain + vec4(time)); });
  const AnalyzedProgram a = analyzeProgram(ast);
  const Reflection& r = a.reflection;
  REQUIRE(r.uniformBlocks.size() == 2);
  CHECK(r.uniformBlocks[0].name == "frameUniforms");
  CHECK(r.uniformBlocks[1].name == "materialUniforms");
  // gain (f32 at 0) sorts before tint (vec4 at 16)
  const StructLayout& l = *r.uniformBlocks[1].layout;
  CHECK(l.find("gain")->offset == 0);
  CHECK(l.find("tint")->offset == 16);
  CHECK(has(compileGLSL(a).fragment, "ub_materialUniforms.tint"));
  CHECK(has(compileWGSL(a).code, "ub_frameUniforms.time"));

  const Val twin = uniform("gain", f32());
  const ProgramAST clash =
      program("c", "C", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(gain + twin)); });
  CHECK_THROWS_AS(analyzeProgram(clash), ShaderError);
}

TEST_CASE("struct and array values") {
  const TypePtr light = makeStruct("Light", {{"dir", vec3f()}, {"power", f32()}});
  const Both b = fragment([&](const Output& out) {
    const Val l = Let(construct(light, {vec3(0.0, 1.0, 0.0), 2.0}), "l");
    const Val weights = Let(array({0.25, 0.5, 0.25}), "weights");
    const Val zero = Var(light, "zero");
    const Val i = Let(toI32(fragCoord().x()) % 3, "i");
    out.set(vec4(l.member("dir") * l.member("power") * weights[i] + zero.member("dir"), 1.0));
  });
  CHECK(has(b.glsl, "struct S_Light {"));
  CHECK(has(b.glsl, "S_Light(vec3(0.0, 1.0, 0.0), 2.0)"));
  CHECK(has(b.glsl, "float[3](0.25, 0.5, 0.25)"));
  // GLSL locals start undefined, so a typed Var is zero filled
  CHECK(has(b.glsl, "S_Light zero_"));
  CHECK(has(b.glsl, " = S_Light(vec3(0), 0.0);"));
  CHECK(has(b.wgsl, "array<f32, 3>(0.25f, 0.5f, 0.25f)"));
  // a let array cannot be indexed at run time in WGSL
  CHECK(has(b.wgsl, "var weights_"));
  CHECK_THROWS_AS(fragment([&](const Output&) { construct(light, {vec3(0.0)}); }), ShaderError);
  CHECK_THROWS_AS(fragment([](const Output&) { array({1.0, vec2(1.0)}); }), ShaderError);
}

TEST_CASE("depth comparison, gradient and offset sampling") {
  const Texture shadow = texture("shadow", TextureDimension::D2,
                                 {.sampleType = TextureSampleType::Depth, .comparison = true});
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Both b = fragment([&](const Output& out) {
    const Val uv = Let(fragCoord().xy() * 0.01, "uv");
    const Val lit = Let(shadow.compare(uv, 0.5), "lit");
    const Val g = Let(albedo.sampleGrad(uv, dpdx(uv), dpdy(uv)), "g");
    const Val o = Let(albedo.sampleOffset(uv, ivec2(1, -1)), "o");
    out.set((g + o) * lit);
  });
  CHECK(has(b.glsl, "uniform highp sampler2DShadow t_shadow;"));
  CHECK(has(b.glsl, "textureLod(t_shadow, vec3(uv_"));
  CHECK(has(b.glsl, "textureGrad(t_albedo, uv_"));
  CHECK(has(b.glsl, "textureLodOffset(t_albedo, uv_"));
  CHECK(has(b.wgsl, "var s_shadow: sampler_comparison;"));
  CHECK(has(b.wgsl, "textureSampleCompareLevel(t_shadow, s_shadow, uv_"));
  CHECK(has(b.wgsl, "textureSampleGrad(t_albedo, s_albedo, uv_"));
  CHECK(has(b.wgsl, ", 0.0f, vec2i(1i, (-1i)))"));
}

TEST_CASE("texture misuse is rejected") {
  const Texture shadow = texture("shadow", TextureDimension::D2,
                                 {.sampleType = TextureSampleType::Depth, .comparison = true});
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Texture sky = texture("sky", TextureDimension::Cube);
  CHECK_THROWS_AS(texture("bad", TextureDimension::D2, {.comparison = true}), ShaderError);
  fragment([&](const Output& out) {
    const Val uv = Let(vec2(0.5), "uv");
    CHECK_THROWS_AS(shadow.sample(uv), ShaderError);
    CHECK_THROWS_AS(shadow.load(ivec2(0, 0)), ShaderError);
    CHECK_THROWS_AS(albedo.compare(uv, 0.5), ShaderError);
    CHECK_THROWS_AS(sky.sampleOffset(vec3(1.0), ivec2(1, 1)), ShaderError);
    const Val dynamic = Let(ivec2(toI32(uv.x()), 0), "dynamic");
    CHECK_THROWS_AS(albedo.sampleOffset(uv, dynamic), ShaderError);
    out.set(vec4(1.0));
  });
}

TEST_CASE("texel offsets are checked after conversion and may be constants") {
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Val offset = constant("OFFSET", ivec2(1, 0));
  const Val tooFar = constant("TOO_FAR", ivec2(9, 0));
  const Both b = fragment([&](const Output& out) {
    const Val uv = Let(vec2(0.5), "uv");
    // the vector constructor truncates 7.9 to 7, which is in -8..7
    const Val a = Let(albedo.sampleOffset(uv, ivec2(vec2(7.9, 0.0))), "a");
    const Val c = Let(albedo.sampleOffset(uv, offset), "c");
    // ~0 is -1; 4 + 3 is 7
    const Val n = Let(albedo.sampleOffset(uv, ~ivec2(0, 0) + ivec2(4, 3)), "n");
    CHECK_THROWS_WITH_AS(albedo.sampleOffset(uv, ivec2(vec2(8.0, 0.0))),
                         doctest::Contains("albedo: texel offset component 8 is not in -8..7"), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, tooFar), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, ivec2(4, 4) + ivec2(4, 4)), ShaderError);
    CHECK_THROWS_AS(albedo.sampleOffset(uv, -ivec2(5, 4) - ivec2(4, 4)), ShaderError);
    out.set(a + c + n);
  });
  CHECK(has(b.glsl, "const ivec2 OFFSET = ivec2(1, 0);"));
  CHECK(has(b.glsl, ", 0.0, OFFSET)"));
  CHECK(has(b.wgsl, ", 0.0f, OFFSET)"));
}

TEST_CASE("textures reflect the sampler binding type and float32-filterable") {
  const Texture color = texture("color", TextureDimension::D2);
  const Texture heights = texture("heights", TextureDimension::D2, {.sampleType = TextureSampleType::UnfilterableFloat, .sampled = true});
  const Texture depth = texture("depth", TextureDimension::D2, {.sampleType = TextureSampleType::Depth, .sampled = true});
  const Texture shadow = texture("shadow", TextureDimension::D2,
                                 {.sampleType = TextureSampleType::Depth, .comparison = true});
  const Texture ids = texture("ids", TextureDimension::D2, {.sampleType = TextureSampleType::Uint});
  const Texture field = texture("field", TextureDimension::D2, {.float32Filterable = true});
  const Output out = output(0);
  const ProgramAST ast = program("t", "T", [] { Position(vec4(0.0)); }, [&] {
    const Val uv = Let(vec2(0.5), "uv");
    out.set(color.sample(uv) + heights.sample(uv) + vec4(depth.sample(uv)) + vec4(shadow.compare(uv, 0.5)) +
            vec4(toF32(ids.load(ivec2(0, 0)).x())) + field.sample(uv));
  });
  const Reflection r = analyzeProgram(ast).reflection;
  const auto sampler = [&](std::string_view name) {
    return std::ranges::find(r.textures, name, &ReflectedTexture::name)->sampler;
  };
  // WebGPU rejects a filtering sampler with unfilterable-float and with depth textures (checked in Chrome 155)
  CHECK(sampler("color") == SamplerBinding::Filtering);
  CHECK(sampler("heights") == SamplerBinding::NonFiltering);
  CHECK(sampler("depth") == SamplerBinding::NonFiltering);
  CHECK(sampler("shadow") == SamplerBinding::Comparison);
  CHECK(sampler("ids") == SamplerBinding::None);
  CHECK(sampler("field") == SamplerBinding::Filtering);
  CHECK(r.deviceFeatures == std::vector{DeviceFeature::Float32Filterable});
  CHECK(deviceFeatureName(DeviceFeature::Float32Filterable) == "float32-filterable");

  CHECK_THROWS_AS(texture("x", TextureDimension::D2, {.sampleType = TextureSampleType::UnfilterableFloat,
                                                      .float32Filterable = true}),
                  ShaderError);
  CHECK_THROWS_AS(texture("y", TextureDimension::D2, {.sampled = false, .float32Filterable = true}), ShaderError);
}

TEST_CASE("dual source blending writes two blend sources at location 0") {
  const Output color = blendOutput(0);
  const Output factor = blendOutput(1);
  const ProgramAST ast = program("d", "D", [] { Position(vec4(0.0)); }, [&] {
    color.set(vec4(1.0, 0.5, 0.25, 1.0));
    factor.set(vec4(0.5));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(a.reflection.deviceFeatures == std::vector{DeviceFeature::DualSourceBlending});
  CHECK(deviceFeatureName(DeviceFeature::DualSourceBlending) == "dual-source-blending");
  REQUIRE(a.reflection.outputs.size() == 2);
  CHECK(a.reflection.outputs[1].blendSrc == 1);
  const std::string w = compileWGSL(a).code;
  CHECK(has(w, "enable dual_source_blending;"));
  CHECK(has(w, "@location(0) @blend_src(0) o_0: vec4f,"));
  CHECK(has(w, "@location(0) @blend_src(1) o_0_src1: vec4f,"));
  CHECK(has(w, "fout.o_0_src1 = vec4f(0.5f);"));
  // WebGL2 needs WEBGL_blend_func_extended
  CHECK_THROWS_WITH_AS(compileGLSL(a), doctest::Contains("d: dual source blending needs WebGPU; WebGL2 cannot run it"), ShaderError);
  const std::string g = compileGLSL(a, {.blendFuncExtended = true}).fragment;
  CHECK(has(g, "#extension GL_EXT_blend_func_extended : require"));
  CHECK(has(g, "layout(location = 0) out vec4 o_0;"));
  CHECK(has(g, "layout(location = 0, index = 1) out vec4 o_0_src1;"));

  // Tint: "@blend_src(1) is missing", "another member with @location doesn't use @blend_src", "same type"
  const Output other = output(1);
  const Output narrow = blendOutput(1, vec3f());
  CHECK_THROWS_AS(analyzeProgram(program("o", "O", [] { Position(vec4(0.0)); }, [&] { color.set(vec4(1.0)); })),
                  ShaderError);
  CHECK_THROWS_AS(analyzeProgram(program("x", "X", [] { Position(vec4(0.0)); }, [&] {
                    color.set(vec4(1.0));
                    factor.set(vec4(1.0));
                    other.set(vec4(1.0));
                  })),
                  ShaderError);
  CHECK_THROWS_WITH_AS(analyzeProgram(program("t", "T", [] { Position(vec4(0.0)); }, [&] {
                         color.set(vec4(1.0));
                         narrow.set(vec3(1.0));
                       })),
                       doctest::Contains("t: the two blend sources have different types, vec4f and vec3f"), ShaderError);
  CHECK_THROWS_AS(blendOutput(2), ShaderError);
}

TEST_CASE("f16 values compile to WGSL with shader-f16 and are rejected by GLSL") {
  CHECK(typeName(f16()) == "f16");
  CHECK(typeName(vec3h()) == "vec3h");
  CHECK(typeName(mat4h()) == "mat4x4h");
  const Output out = output(0);
  const ProgramAST ast = program("h", "H", [] { Position(vec4(0.0)); }, [&] {
    const Val c = Let(hvec4(1.0, 0.5, 0.25, 1.0) * 2.0, "c");
    const Val n = Let(length(c.xyz()), "n");
    const Val m = Let(hmat2(2.0) * hvec2(1.0, 0.5), "m");
    out.set(construct(vec4f(), {c * n + hvec4(m, 0.0, 0.0)}));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(a.reflection.deviceFeatures == std::vector{DeviceFeature::ShaderF16});
  CHECK(deviceFeatureName(DeviceFeature::ShaderF16) == "shader-f16");
  const std::string w = compileWGSL(a).code;
  CHECK(has(w, "enable f16;"));
  CHECK(has(w, "vec4h(1.0h, 0.5h, 0.25h, 1.0h) * 2.0h"));
  // length of an f16 vector is f16; the diagonal matrix is written out with f16 zeros
  CHECK(has(w, ": f16 = length("));
  CHECK(has(w, "mat2x2h(2.0h, 0.0h, 0.0h, 2.0h)"));
  CHECK_THROWS_WITH_AS(compileGLSL(a), doctest::Contains("h: f16 needs WebGPU; WebGL2 cannot run it"), ShaderError);

  program("e", "E", [] { Position(vec4(0.0)); }, [&] {
    const Val x = Let(hvec2(0.5), "x");
    // WGSL has no f16 derivatives or f16 packing
    CHECK_THROWS_WITH_AS(dpdx(x), doctest::Contains("dpdx needs an f32 scalar or vector, got vec2h"), ShaderError);
    CHECK_THROWS_AS(packHalf2x16(x), ShaderError);
    CHECK_THROWS_WITH_AS(toF16(70000.0), doctest::Contains("Literal 70000 does not fit in f16"), ShaderError);
    // f16 and f32 do not mix without a conversion
    CHECK_THROWS_AS(x + vec2(1.0), ShaderError);
    // the f16 step above 1 is 2^-10, so 1 + 0.0001 rounds to 1: equal edges
    CHECK_THROWS_AS(smoothstep(toF16(1.0), toF16(1.0) + toF16(0.0001), x), ShaderError);
    CHECK(smoothstep(toF16(1.0), toF16(1.0) + toF16(1.0 / 1024.0), x).type() == vec2h());
    out.set(vec4(1.0));
  });
}

TEST_CASE("f16 uniform fields follow the WGSL layout and the writer stores binary16") {
  const UniformBlock block = uniformBlock(
      "halves", {{"a", f16()}, {"b", vec3h()}, {"c", f32()}, {"d", vec2h()}, {"e", mat2h()}, {"f", mat3h()}});
  // WGSL alignment and size: f16 2/2, vec2h 4/4, vec3h 8/6, mat2x2h 4/8, mat3x3h 8/24
  const auto l = structLayout(block.resource()->structType);
  const auto offset = [&](std::string_view f) { return l->find(f)->offset; };
  CHECK(offset("a") == 0);
  CHECK(offset("b") == 8);
  CHECK(offset("c") == 16);
  CHECK(offset("d") == 20);
  CHECK(offset("e") == 24);
  CHECK(offset("f") == 32);
  CHECK(l->size == 64);
  CHECK(storageLayout(vec3h()).stride == 8);
  CHECK(storageLayout(mat4h()).size == 32);

  UniformWriter w(l);
  w.setFloat("a", 1.0f);
  const std::array<float, 3> b{65504.0f, 0.1f, -2.0f};
  w.setVec("b", b);
  const auto half = [&](std::size_t at) {
    std::uint16_t v;
    std::memcpy(&v, w.bytes().data() + at, 2);
    return v;
  };
  // binary16: 1.0 is 0x3c00, 65504 is 0x7bff, 0.1 rounds to 0x2e66, -2 is 0xc000
  CHECK(half(0) == 0x3c00);
  CHECK(half(8) == 0x7bff);
  CHECK(half(10) == 0x2e66);
  CHECK(half(12) == 0xc000);

  // a declared f16 field needs shader-f16 even when the shader never reads it
  const UniformBlock mixed = uniformBlock("mixed", {{"scale", f32()}, {"unused", f16()}});
  const Output out = output(0);
  const ProgramAST ast = program("u", "U", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(mixed["scale"])); });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(a.reflection.deviceFeatures == std::vector{DeviceFeature::ShaderF16});
  CHECK(has(compileWGSL(a).code, "enable f16;"));
}

TEST_CASE("round() rounds half to even in both backends") {
  const Both b = fragment([](const Output& out) {
    out.set(vec4(round(fragCoord().xy() * 0.5), 0.0, 1.0));
  });
  // WGSL round() rounds half to even; GLSL ES 3.00 round() may round .5 either way, roundEven() matches WGSL
  CHECK(has(b.glsl, "roundEven(("));
  CHECK_FALSE(has(b.glsl, " round("));
  CHECK(has(b.wgsl, "round(("));
}

TEST_CASE("f32 and f16 matrices get their own WGSL inverse and mat3 helpers") {
  const Function pick = fn("pick", {}, mat4h(), [](const Args&) { Return(hmat4(1.0)); });
  const Function pickF = fn("pickF", {}, mat4f(), [](const Args&) { Return(mat4(1.0)); });
  const Output out = output(0);
  const ProgramAST ast = program("i", "I", [] { Position(vec4(0.0)); }, [&] {
    const Val h2 = Let(inverse(hmat2(2.0)), "h2");
    const Val f2 = Let(inverse(mat2(2.0)), "f2");
    const Val h3 = Let(inverse(hmat3(2.0)), "h3");
    const Val h4 = Let(inverse(hmat4(2.0)), "h4");
    // a call operand makes the 4x4 to 3x3 conversion a helper, so it runs once
    const Val c3 = Let(construct(mat3h(), {pick()}), "c3");
    const Val d3 = Let(construct(mat3f(), {pickF()}), "d3");
    const Val k3 = Let(construct(mat3h(), {h4}), "k3");
    // the same shape converts between f32 and f16
    const Val back = Let(construct(mat2f(), {h2}), "back");
    out.set(vec4(construct(vec2f(), {(h2 * hvec2(1.0)) + hvec2(h3[0].xy()) + hvec2(c3[0].xy()) + hvec2(k3[1].xy())}) +
                     (f2 + back) * vec2(1.0) + d3[0].xy(), 0.0, 1.0));
  });
  const std::string w = compileWGSL(analyzeProgram(ast)).code;
  // one helper per matrix type, and each returns its own type
  CHECK(has(w, "fn moga_inverse_mat2x2h(m: mat2x2h) -> mat2x2h {"));
  CHECK(has(w, "return mat2x2h(m[1][1], -m[0][1], -m[1][0], m[0][0]) * (1.0 / det);"));
  CHECK(has(w, "fn moga_inverse_mat2x2f(m: mat2x2f) -> mat2x2f {"));
  CHECK(has(w, "return transpose(mat3x3h(cross(b, c), cross(c, a), cross(a, b))) * (1.0 / det);"));
  CHECK(has(w, "fn moga_inverse_mat4x4h(m: mat4x4h) -> mat4x4h {"));
  CHECK(has(w, "return mat4x4h("));
  CHECK(has(w, "fn moga_mat3x3h_from_mat4x4h(m: mat4x4h) -> mat3x3h { return mat3x3h(m[0].xyz, m[1].xyz, m[2].xyz); }"));
  CHECK(has(w, "fn moga_mat3x3f_from_mat4x4f(m: mat4x4f) -> mat3x3f {"));
  // without side effects the 3x3 is written out with its own type
  CHECK(has(w, "mat3x3h(h4_"));
  CHECK(has(w, "mat2x2f(h2_"));
}

TEST_CASE("GLSL converts an unsigned texture level to int") {
  const Texture albedo = texture("albedo", TextureDimension::D2);
  const Both b = fragment([&](const Output& out) {
    const Val level = Let(toU32(fragCoord().x()) % 2u, "level");
    const Val s = Let(albedo.size(toU32(0)), "s");
    const Val t = Let(albedo.load(ivec2(0, 0), level), "t");
    const Val i = Let(albedo.load(ivec2(1, 1), 0), "i");
    out.set(t + i + vec4(toF32(s.x())));
  });
  // GLSL ES 3.00 texelFetch and textureSize take an int lod
  CHECK(has(b.glsl, "textureSize(t_albedo, int(0u))"));
  CHECK(has(b.glsl, "texelFetch(t_albedo, ivec2(0, 0), int(level_"));
  // an i32 level stays as it is; WGSL takes either type
  CHECK(has(b.glsl, "texelFetch(t_albedo, ivec2(1, 1), 0)"));
  CHECK(has(b.wgsl, "textureDimensions(t_albedo, 0u)"));
}
