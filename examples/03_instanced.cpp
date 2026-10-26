// Instanced draw reading per instance transforms from a table, a bounded loop,
// reflection output and a uniform staging buffer.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <array>
#include <print>

using namespace moga;

namespace {

std::string_view kindName(const ReflectedResource& r) {
  if (std::holds_alternative<ReflectedUniformBlock>(r)) return "uniform";
  if (std::holds_alternative<ReflectedTexture>(r)) return "texture";
  return "table";
}

std::string_view resourceName(const ReflectedResource& r) {
  return std::visit([](const auto& x) -> std::string_view { return x.name; }, r);
}

} // namespace

int main() {
  Stopwatch sw;
  const UniformBlock frame = uniformBlock(
      "frame", {{"viewProj", mat4f()}, {"time", f32()}, {"waves", i32()}},
      ResourceFrequency::Frame);
  const Table instances =
      table("instances", {{"model", mat4f()}, {"tint", vec4f()}}, ResourceFrequency::Object);

  const Val position = attribute("position", vec3f());
  const Varying vTint = varying("tint", vec4f());
  const Varying vId = varying("id", u32(), Interpolation::FlatEither);
  const Output fragColor = output(0);

  const BuildResult result = buildShader(
      "instanced", "INSTANCED",
      [&] {
        const TableRecord record = instances.at(instanceIndex());
        const Val offset = Var(0.0, "offset");
        For(0, frame["waves"], [&](const Val& i) {
          offset.addAssign(sin(frame["time"] + toF32(i) * position.x()) * 0.1);
        });
        const Val world = Let(record["model"] * vec4(position + vec3(0.0, offset, 0.0), 1.0), "world");
        vTint.set(record["tint"]);
        vId.set(instanceIndex());
        Position(frame["viewProj"] * world);
      },
      [&] {
        If((vId.value() % 2u).eq(0u), [&] { Discard(); });
        fragColor.set(vTint.value());
      });

  sw.lap("build");
  if (!report(result)) return 1;
  printCode(result);
  sw.print(result.program->reflection.variantKey);

  const Reflection& r = result.program->reflection;
  std::println("reflection for {}:", r.variantKey);
  for (std::size_t g = 0; g < r.groups.size(); ++g)
    for (const ReflectedResource& res : r.groups[g])
      std::println("  group {} binding {}: {} {}", g, bindingOf(res), kindName(res), resourceName(res));
  if (result.glsl->baseInstanceUniform) std::println("  GLSL base instance uniform: {}", *result.glsl->baseInstanceUniform);

  const StructLayout& layout = *r.uniformBlocks.front().layout;
  UniformWriter writer(r.uniformBlocks.front().layout);
  std::array<float, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  writer.setMat4("viewProj", identity);
  writer.setFloat("time", 1.5f);
  writer.setInt("waves", 3);
  std::println("frame block: {} bytes", writer.bytes().size());
  for (const FieldLayout& f : layout.fields)
    std::println("  {} at {} ({} bytes)", f.name, f.offset, f.size);
}
