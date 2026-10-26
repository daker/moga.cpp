#include <doctest/doctest.h>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

#include <array>
#include <cstring>

using namespace moga;

// Expected offsets are worked out by hand from the std140 rules.

TEST_CASE("std140 offsets for mixed members") {
  const TypePtr s =
      makeStruct("Mixed", {{"a", f32()}, {"b", vec3f()}, {"c", mat4f()}, {"d", vec2f()}});
  const auto l = structLayout(s);
  CHECK(l->find("a")->offset == 0);
  CHECK(l->find("b")->offset == 16);
  CHECK(l->find("c")->offset == 32);
  CHECK(l->find("d")->offset == 96);
  CHECK(l->size == 112);
}

TEST_CASE("a scalar packs into the tail of a vec3") {
  const auto l = structLayout(makeStruct("Packed", {{"dir", vec3f()}, {"w", f32()}}));
  CHECK(l->find("w")->offset == 12);
  CHECK(l->size == 16);
}

TEST_CASE("members after an array start on a 16 byte boundary") {
  const auto l =
      structLayout(makeStruct("Arr", {{"items", arrayOf(vec4f(), 3)}, {"n", u32()}}));
  CHECK(l->find("n")->offset == 48);
  CHECK(l->size == 64);
}

TEST_CASE("mat3 occupies three padded columns") {
  const auto l = structLayout(makeStruct("M3", {{"m", mat3f()}, {"x", f32()}}));
  CHECK(l->find("x")->offset == 48);
}

TEST_CASE("layouts that differ between std140 and WGSL are rejected") {
  CHECK_THROWS_AS(structLayout(makeStruct("B", {{"flag", boolType()}})), ShaderError);
  CHECK_THROWS_AS(structLayout(makeStruct("M2", {{"m", mat2f()}})), ShaderError);
  CHECK_THROWS_AS(structLayout(makeStruct("F", {{"f", arrayOf(f32(), 4)}})), ShaderError);
}

TEST_CASE("uniform writer pads mat3 columns") {
  const auto l = structLayout(makeStruct("W", {{"m", mat3f()}}));
  UniformWriter w(l);
  const std::array<float, 9> m{1, 2, 3, 4, 5, 6, 7, 8, 9};
  w.setMat3("m", m);
  std::array<float, 12> out{};
  std::memcpy(out.data(), w.bytes().data(), sizeof out);
  CHECK(out == std::array<float, 12>{1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0});
  CHECK_THROWS_AS(w.setFloat("missing", 1.0f), ShaderError);
}

TEST_CASE("table field offsets count texels") {
  const auto off = tableFieldOffsets(
      makeStruct("R", {{"model", mat4f()}, {"tint", vec4f()}, {"extra", vec4f()}}));
  CHECK(off.at("model") == 0);
  CHECK(off.at("tint") == 4);
  CHECK(off.at("extra") == 5);
}
