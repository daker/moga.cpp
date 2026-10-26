#include <doctest/doctest.h>
#include <bit>
#include <cstdint>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

using namespace moga;

TEST_CASE("vector and matrix types are interned") {
  CHECK(vectorType(ScalarKind::F32, 3) == vec3f());
  CHECK(vectorType(ScalarKind::U32, 1) == u32());
  CHECK(squareMatrix(4) == mat4f());
}

TEST_CASE("structs compare by name, arrays by element and length") {
  const TypePtr a = makeStruct("Light", {{"color", vec4f()}});
  const TypePtr b = makeStruct("Light", {{"color", vec4f()}});
  CHECK(a != b);
  CHECK(typesEqual(a, b));
  CHECK(typesEqual(arrayOf(vec4f(), 4), arrayOf(vec4f(), 4)));
  CHECK_FALSE(typesEqual(arrayOf(vec4f(), 4), arrayOf(vec4f(), 3)));
}

TEST_CASE("type names follow WGSL spelling") {
  CHECK(typeName(vec3f()) == "vec3f");
  CHECK(typeName(vec2b()) == "vec2<bool>");
  CHECK(typeName(mat3f()) == "mat3x3f");
  CHECK(typeName(arrayOf(vec4u(), 8)) == "array<vec4u, 8>");
}

TEST_CASE("variant keys are sorted by option name") {
  const VariantOptions options{
      {"tex", true}, {"clip", 2.0}, {"lighting", std::string("phong")}, {"pick", false}};
  CHECK(variantKey("polydata", options) ==
        "polydata|clip=2|lighting=\"phong\"|pick=false|tex=true");
}

TEST_CASE("different option sets never share a variant key") {
  const auto key = [](VariantOptions o) { return variantKey("p", o); };
  // the same text in three types
  CHECK(key({{"x", true}}) != key({{"x", 1.0}}));
  CHECK(key({{"x", 1.0}}) != key({{"x", std::string("1")}}));
  CHECK(key({{"x", true}}) != key({{"x", std::string("true")}}));
  // case is kept
  CHECK(key({{"x", std::string("phong")}}) != key({{"x", std::string("PHONG")}}));
  CHECK(key({{"tex", true}}) != key({{"TEX", true}}));
  // a string that looks like more options
  CHECK(key({{"a", std::string("1\"|b=\"2")}}) != key({{"a", std::string("1")}, {"b", std::string("2")}}));
  CHECK(key({{"a|b", true}}) != key({{"a", true}, {"b", true}}));
}

TEST_CASE("NaN options with different bits have different variant keys") {
  const auto key = [](VariantOptions o) { return variantKey("p", o); };
  // two quiet NaNs that differ only in the lowest payload bit
  const double a = std::bit_cast<double>(std::uint64_t{0x7FF8000000000000});
  const double b = std::bit_cast<double>(std::uint64_t{0x7FF8000000000001});
  CHECK(key({{"x", a}}) != key({{"x", b}}));
  CHECK(key({{"x", a}}) == key({{"x", a}}));
}
