#include <doctest/doctest.h>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

using namespace moga;

namespace {
/** Runs `body` inside a vertex stage with a trivial fragment stage. */
template <class F> ProgramAST inVertex(F&& body) {
  return program(
      "t", "T",
      [&] {
        body();
        Position(vec4(0.0));
      },
      [] {});
}
} // namespace

TEST_CASE("matrix products follow linear algebra shapes") {
  inVertex([] {
    const Val m4 = Var(mat4f());
    const Val m3 = Var(mat3f());
    CHECK((m4 * vec4(1.0)).type() == vec4f());
    CHECK((vec3(1.0) * m3).type() == vec3f());
    CHECK((m4 * m4).type() == mat4f());
    CHECK((m4 * 2).type() == mat4f());
    CHECK_THROWS_AS(m4 * vec3(1.0), ShaderError);
    CHECK_THROWS_AS(m4 / m4, ShaderError);
  });
}

TEST_CASE("number operands take the type of the shader value") {
  inVertex([] {
    const Val i = Var(toI32(3));
    CHECK((i + 1).type() == i32());
    CHECK((2 * vec3(1.0)).type() == vec3f());
    CHECK_THROWS_AS(i + 0.5, ShaderError);
    CHECK((vec4(1.0) < vec4(2.0)).type() == vec4b());
    CHECK((toU32(1) << toI32(2)).type() == u32());
  });
}

TEST_CASE("mismatched operands are rejected") {
  inVertex([] {
    CHECK_THROWS_AS(vec3(1.0) + vec2(1.0), ShaderError);
    CHECK_THROWS_AS(toI32(1) + toU32(1), ShaderError);
    CHECK_THROWS_AS(!vec2(1.0), ShaderError);
  });
}

TEST_CASE("swizzles accept color names and check range") {
  inVertex([] {
    const Val v = Var(vec3(1.0));
    CHECK(v.swizzle("bgr").type() == vec3f());
    CHECK_THROWS_AS(v.swizzle("w"), ShaderError);
    CHECK_THROWS_AS(v.swizzle("xq"), ShaderError);
  });
}

TEST_CASE("constructors check component counts") {
  inVertex([] {
    CHECK_NOTHROW(vec4(vec2(1.0), 0.0, 1.0));
    CHECK_NOTHROW(vec3(1.0));
    CHECK_NOTHROW(mat3(Var(mat4f())));
    CHECK_THROWS_AS(vec4(vec2(1.0), 0.0), ShaderError);
  });
}

TEST_CASE("immutable bindings cannot be assigned") {
  inVertex([] {
    const Val t = Let(1.0);
    CHECK_THROWS_AS(t.set(2.0), ShaderError);
    const Val v = Var(vec3(0.0));
    CHECK_NOTHROW(v.x().set(1.0));
  });
}

TEST_CASE("stage rules are enforced") {
  const Varying vary = varying("c", vec3f());
  CHECK_THROWS_AS(inVertex([&] { (void)vary.value(); }), ShaderError);
  CHECK_THROWS_AS(inVertex([] { Discard(); }), ShaderError);
  CHECK_THROWS_AS(varying("id", u32()), ShaderError);
  CHECK_THROWS_AS(Var(1.0), ShaderError);
  CHECK_FALSE(inStage().has_value());
}

TEST_CASE("tables accept only whole texel fields") {
  CHECK_THROWS_AS(table("bad", {{"f", f32()}}), ShaderError);
  CHECK(table("ok", {{"m", mat4f()}, {"c", vec4f()}}).resource()->stride == 5);
}

TEST_CASE("a function rejects repeated parameter names") {
  CHECK_THROWS_WITH_AS(fn("f", {{"x", f32()}, {"x", f32()}}, f32(), [](const Args& a) { Return(a["x"]); }),
                       doctest::Contains("Function f: parameter x is declared twice"), ShaderError);
  // distinct names stay accessible by name
  const Function g = fn("g", {{"x", f32()}, {"y", f32()}}, f32(), [](const Args& a) { Return(a["x"] - a["y"]); });
  CHECK(g.decl()->params.size() == 2);
}

TEST_CASE("integer remainder needs matching vector sizes") {
  const Output out = output(0);
  program("r", "R", [] { Position(vec4(0.0)); }, [&] {
    const Val a = Let(ivec2(7, 9), "a");
    const Val b = Let(ivec3(2, 4, 5), "b");
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Rem, a, b), doctest::Contains("Invalid operands for \"%\": vec2i and vec3i"), ShaderError);
    CHECK_THROWS_AS(binary(BinaryOp::Rem, b, a), ShaderError);
    // same size, and a scalar with a vector, stay valid in both languages
    CHECK(binary(BinaryOp::Rem, a, ivec2(4, 5)).type() == vec2i());
    CHECK(binary(BinaryOp::Rem, a, 4).type() == vec2i());
    CHECK(binary(BinaryOp::Rem, toI32(7), b).type() == vec3i());
    out.set(vec4(1.0));
  });
}

TEST_CASE("integer division and remainder by a constant zero are rejected") {
  const Val zero = constant("ZERO_DIVISOR", toI32(2) - toI32(2));
  const Output out = output(0);
  program("z", "Z", [&] {
    const Val x = Let(toI32(vertexIndex()), "x");
    const Val v = Let(ivec2(x, x), "v");
    // Tint (Chrome 155) reports "integer division by zero is invalid" for each of these
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Div, x, 0), doctest::Contains("Integer \"/\" by a constant zero"), ShaderError);
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Rem, x, 0), doctest::Contains("Integer \"%\" by a constant zero"), ShaderError);
    CHECK_THROWS_AS(v / ivec2(1, 0), ShaderError);
    CHECK_THROWS_AS(v % 0, ShaderError);
    CHECK_THROWS_AS(x / zero, ShaderError);
    CHECK_THROWS_AS(x / min(toI32(0), toI32(3)), ShaderError);
    const Val counter = Var(x, "counter");
    CHECK_THROWS_AS(counter.divAssign(0), ShaderError);
    CHECK_THROWS_AS(counter.remAssign(0), ShaderError);
    // valid: a non zero constant, a divisor known only at run time, and float division
    CHECK(binary(BinaryOp::Div, x, 2).type() == i32());
    CHECK(binary(BinaryOp::Div, x, x).type() == i32());
    CHECK(binary(BinaryOp::Div, toF32(x), 0.0).type() == f32());
    Position(vec4(0.0));
  }, [&] { out.set(vec4(1.0)); });
}

TEST_CASE("an index known when the shader is built must be in range") {
  const Val second = constant("SECOND", toU32(1) + toU32(1));
  const Output out = output(0);
  program("i", "I", [&] {
    const Val x = Let(toF32(vertexIndex()), "x");
    const Val v = Var(vec2(x, x), "v");
    const Val m = Var(mat3(1.0), "m");
    const Val a = Var(arrayOf(f32(), 4), "a");
    // Tint (Chrome 155): "index 2 out of bounds [0..1]", and likewise for each case below
    CHECK_THROWS_WITH_AS(v[2], doctest::Contains("Index 2 is out of range for vec2f (0..1)"), ShaderError);
    CHECK_THROWS_AS(v[-1], ShaderError);
    CHECK_THROWS_WITH_AS(m[3], doctest::Contains("Index 3 is out of range for mat3x3f (0..2)"), ShaderError);
    CHECK_THROWS_AS(a[4], ShaderError);
    CHECK_THROWS_AS(v[second], ShaderError);
    // in range, and an index known only at run time
    CHECK(v[1].type() == f32());
    CHECK(m[2].type() == vec3f());
    CHECK(a[3].type() == f32());
    CHECK(a[vertexIndex() + 9u].type() == f32());
    Position(vec4(0.0));
  }, [&] { out.set(vec4(1.0)); });
}

TEST_CASE("min and max widen a scalar on either side, and sign takes i32") {
  const Output out = output(0);
  program("m", "M", [&] {
    const Val v = Let(vec3(1.0, 2.0, 3.0), "v");
    CHECK(min(v, 1.0).type() == vec3f());
    CHECK(min(1.0, v).type() == vec3f());
    CHECK(max(toF32(vertexIndex()), v).type() == vec3f());
    // the scalar must have the vector's kind
    CHECK_THROWS_AS(min(toI32(1), v), ShaderError);
    // WGSL and GLSL ES 3.00 define sign() for signed integers
    CHECK(sign(ivec3(-2, 0, 5)).type() == vec3i());
    CHECK(sign(toI32(vertexIndex())).type() == i32());
    CHECK_THROWS_AS(sign(toU32(vertexIndex())), ShaderError);
    Position(vec4(0.0));
  }, [&] { out.set(vec4(1.0)); });
}

TEST_CASE("float constant expressions that are infinite, NaN or outside a builtin's domain are rejected") {
  const Val inf = constant("TOO_BIG", 3.0e38);
  const Output out = output(0);
  program("f", "F", [&] {
    const Val x = Let(toF32(vertexIndex()), "x");
    // Tint (Chrome 155) rejects each of these at shader creation
    CHECK_THROWS_WITH_AS(binary(BinaryOp::Div, 1.0, 0.0),
                         doctest::Contains("\"/\": the constant value is infinite in f32, which WGSL rejects"), ShaderError);
    CHECK_THROWS_AS(binary(BinaryOp::Div, 0.0, 0.0), ShaderError);
    CHECK_THROWS_AS(inf * 10.0, ShaderError);
    CHECK_THROWS_AS(inf + inf, ShaderError);
    CHECK_THROWS_AS(vec2(1.0, 1.0) / vec2(1.0, 0.0), ShaderError);
    CHECK_THROWS_AS(toF16(60000.0) + toF16(60000.0), ShaderError);
    CHECK_THROWS_AS(construct(f16(), {toF32(70000.0)}), ShaderError);
    CHECK_THROWS_WITH_AS(inverseSqrt(toF32(0.0)), doctest::Contains("inverseSqrt: the constant argument 0 is outside its domain"), ShaderError);
    CHECK_THROWS_AS(log2(toF32(0.0)), ShaderError);
    CHECK_THROWS_AS(asin(toF32(1.5)), ShaderError);
    CHECK_THROWS_AS(acosh(toF32(0.5)), ShaderError);
    CHECK_THROWS_AS(atanh(toF32(1.0)), ShaderError);
    CHECK_THROWS_AS(pow(toF32(-1.0), 2.0), ShaderError);
    CHECK_THROWS_AS(pow(toF32(0.0), 0.0), ShaderError);
    CHECK_THROWS_AS(exp(toF32(100.0)), ShaderError);
    CHECK_THROWS_AS(degrees(inf), ShaderError);
    CHECK_THROWS_AS(normalize(vec2(0.0)), ShaderError);
    CHECK_THROWS_AS(length(vec2(inf)), ShaderError);
    // -2147483648 / -1 is the one i32 quotient that overflows
    const Val minI32 = toI32(-2147483647) - toI32(1);
    CHECK_THROWS_WITH_AS(minI32 / toI32(-1), doctest::Contains("Integer \"/\" of -2147483648 by -1 overflows i32"), ShaderError);
    CHECK_THROWS_AS(minI32 % toI32(-1), ShaderError);
    CHECK_NOTHROW(toI32(-2147483647) / toI32(-1));
    // every product, square and sum of these is a step that can overflow (Tint, Chrome 155)
    CHECK_THROWS_AS(pow(toF32(2.0), 128.0), ShaderError);
    CHECK_NOTHROW(pow(toF32(2.0), 127.0));
    CHECK_THROWS_AS(pow(toF32(10.0), 39.0), ShaderError);
    CHECK_THROWS_AS(length(vec2(1.0e20, 0.0)), ShaderError);
    CHECK_THROWS_AS(normalize(vec2(1.0e20, 0.0)), ShaderError);
    CHECK_THROWS_AS(distance(vec2(1.0e20, 0.0), vec2(0.0)), ShaderError);
    CHECK_THROWS_AS(dot(vec2(2.0e19, 1.0), vec2(2.0e19, 1.0)), ShaderError);
    CHECK_NOTHROW(dot(vec2(1.0e19, 1.0e19), vec2(1.0e19, 1.0e19)));
    CHECK_THROWS_AS(cross(vec3(3.0e38, 0.0, 0.0), vec3(0.0, 3.0e38, 0.0)), ShaderError);
    CHECK_NOTHROW(cross(vec3(1.0e20, 0.0, 0.0), vec3(1.0e20, 0.0, 0.0)));
    CHECK_THROWS_AS(mix(toF32(3.0e38), -3.0e38, 3.0), ShaderError);
    CHECK_NOTHROW(mix(toF32(1.0e38), 1.0e38, 0.5));
    CHECK_THROWS_AS(reflect(vec2(1.0e20, 0.0), vec2(1.0e20, 0.0)), ShaderError);
    // plain double arguments would call the C library; toF32 makes them shader values
    // and accepts these: the boundaries of each domain, and run time values
    CHECK_NOTHROW(sqrt(toF32(0.0)));
    CHECK_NOTHROW(asin(toF32(1.0)));
    CHECK_NOTHROW(acosh(toF32(1.0)));
    CHECK_NOTHROW(atanh(toF32(0.5)));
    CHECK_NOTHROW(atan2(toF32(0.0), 0.0));
    CHECK_NOTHROW(x / 0.0);
    CHECK_NOTHROW(log(x));
    Position(vec4(0.0));
  }, [&] { out.set(vec4(1.0)); });
}

TEST_CASE("a ShaderError names the C++ line and column that was building the program") {
  const Output out = output(0);
  const auto where = [](const std::function<void()>& f) {
    try {
      f();
    } catch (const ShaderError& e) {
      return std::pair(e.where(), std::string(e.what()));
    }
    return std::pair(std::optional<DebugInfo>{}, std::string());
  };
  program("w", "W", [&] {
    const Val x = Let(toF32(vertexIndex()), "x");
    // the error names the operand at fault: the literal 1e52
    const int line = __LINE__; auto [at, what] = where([&] { (void)(x + 1e52); });
    REQUIRE(at.has_value());
    CHECK(at->line == static_cast<std::uint32_t>(line));
    CHECK(at->column == 73u);
    CHECK(at->file.ends_with("dsl.test.cpp"));
    CHECK(what.find(std::format("dsl.test.cpp:{}:", line)) != std::string::npos);
    CHECK(what.ends_with("Literal 1e+52 does not fit in f32"));
    // an error found inside a library helper still names the user's call
    const int helperLine = __LINE__; auto [helperAt, helperWhat] = where([&] { (void)clamp(x, 1.0, 0.0); });
    REQUIRE(helperAt.has_value());
    CHECK(helperAt->line == static_cast<std::uint32_t>(helperLine));
    Position(vec4(0.0));
  }, [&] { out.set(vec4(1.0)); });
  // an error about the whole program carries no stale C++ location
  try {
    analyzeProgram(program("p", "P", [] {}, [&] { out.set(vec4(1.0)); }));
    FAIL("expected an error");
  } catch (const ShaderError& e) {
    CHECK_FALSE(e.where().has_value());
    CHECK(std::string(e.what()) == e.message());
  }
}
