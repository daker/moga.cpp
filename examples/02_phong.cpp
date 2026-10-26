// Textured Blinn-Phong with a frame uniform block, a material block, a helper
// function and a variant that toggles the texture.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <print>

using namespace moga;

namespace {

BuildResult buildPhong(bool textured) {
  const UniformBlock camera = uniformBlock(
      "camera", {{"viewProj", mat4f()}, {"eye", vec4f()}}, ResourceFrequency::Frame);
  const UniformBlock material = uniformBlock(
      "material", {{"diffuse", vec4f()}, {"specular", vec4f()}, {"shininess", f32()}});
  const Texture albedo = texture("albedo", TextureDimension::D2);

  const Val position = attribute("position", vec3f());
  const Val normal = attribute("normal", vec3f());
  const Val uv = attribute("uv", vec2f());
  const Varying vNormal = varying("normal", vec3f());
  const Varying vWorld = varying("world", vec3f());
  const Varying vUv = varying("uv", vec2f());
  const Output fragColor = output(0);

  const Function blinn = fn("blinn", {{"n", vec3f()}, {"l", vec3f()}, {"v", vec3f()}, {"p", f32()}},
                            f32(), [](const Args& a) {
                              const Val h = Let(normalize(a["l"] + a["v"]), "h");
                              Return(pow(max(dot(a["n"], h), 0.0), a["p"]));
                            });

  const VariantOptions options{{"tex", textured}};
  return buildShader(
      "phong", variantKey("phong", options),
      [&] {
        vNormal.set(normal);
        vWorld.set(position);
        vUv.set(uv);
        Position(camera["viewProj"] * vec4(position, 1.0));
      },
      [&] {
        const Val n = Let(normalize(vNormal.value()), "n");
        const Val v = Let(normalize(camera["eye"].xyz() - vWorld.value()), "v");
        const Val l = Let(normalize(vec3(0.3, 1.0, 0.5)), "l");
        const Val base = Var(material["diffuse"].rgb(), "base");
        if (textured) base.mulAssign(albedo.sample(vUv.value()).rgb());
        const Val color = Var(base * max(dot(n, l), 0.0), "color");
        If(frontFacing(), [&] {
          color.addAssign(material["specular"].rgb() * blinn(n, l, v, material["shininess"]));
        }).Else([&] { color.set(color * 0.5); });
        fragColor.set(vec4(color, 1.0));
      });
}

} // namespace

int main() {
  for (const bool textured : {false, true}) {
    Stopwatch sw;
    const BuildResult result = buildPhong(textured);
    sw.lap("build");
    if (!report(result)) return 1;
    std::println("// ==== {}", result.program->reflection.variantKey);
    printCode(result);
    sw.print(result.program->reflection.variantKey);
  }
}
