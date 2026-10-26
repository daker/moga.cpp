// Metallic roughness PBR: GGX distribution, Smith geometry and Schlick
// Fresnel for one directional light, plus a constant ambient term.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <numbers>
#include <print>

using namespace moga;

int main() {
  Stopwatch sw;
  const UniformBlock camera = uniformBlock(
      "camera", {{"viewProj", mat4f()}, {"eye", vec4f()}}, ResourceFrequency::Frame);
  const UniformBlock light = uniformBlock(
      "light", {{"direction", vec4f()}, {"color", vec4f()}}, ResourceFrequency::Frame);
  // metallicRoughness: x metallic, y roughness
  const UniformBlock material =
      uniformBlock("material", {{"baseColor", vec4f()}, {"metallicRoughness", vec4f()}});

  const Val position = attribute("position", vec3f());
  const Val normal = attribute("normal", vec3f());
  const Varying vNormal = varying("normal", vec3f());
  const Varying vWorld = varying("world", vec3f());
  const Output fragColor = output(0);

  const Function distributionGGX =
      fn("distributionGGX", {{"nh", f32()}, {"roughness", f32()}}, f32(), [](const Args& a) {
        const Val alpha = Let(a["roughness"] * a["roughness"], "alpha");
        const Val a2 = Let(alpha * alpha, "a2");
        const Val d = Let(a["nh"] * a["nh"] * (a2 - 1.0) + 1.0, "d");
        Return(a2 / (std::numbers::pi * d * d));
      });

  const Function geometrySchlick =
      fn("geometrySchlick", {{"nx", f32()}, {"k", f32()}}, f32(),
         [](const Args& a) { Return(a["nx"] / (a["nx"] * (1.0 - a["k"]) + a["k"])); });

  const Function geometrySmith =
      fn("geometrySmith", {{"nv", f32()}, {"nl", f32()}, {"roughness", f32()}}, f32(),
         [&](const Args& a) {
           const Val r = Let(a["roughness"] + 1.0, "r");
           const Val k = Let(r * r / 8.0, "k");
           Return(geometrySchlick(a["nv"], k) * geometrySchlick(a["nl"], k));
         });

  const Function fresnelSchlick =
      fn("fresnelSchlick", {{"cosTheta", f32()}, {"f0", vec3f()}}, vec3f(), [](const Args& a) {
        Return(a["f0"] + (1.0 - a["f0"]) * pow(clamp(1.0 - a["cosTheta"], 0.0, 1.0), 5.0));
      });

  const BuildResult result = buildShader(
      "pbr", variantKey("pbr", {}),
      [&] {
        vNormal.set(normal);
        vWorld.set(position);
        Position(camera["viewProj"] * vec4(position, 1.0));
      },
      [&] {
        const Val albedo = Let(material["baseColor"].rgb(), "albedo");
        const Val metallic = Let(material["metallicRoughness"].x(), "metallic");
        const Val roughness = Let(max(material["metallicRoughness"].y(), 0.04), "roughness");

        const Val n = Let(normalize(vNormal.value()), "n");
        const Val v = Let(normalize(camera["eye"].xyz() - vWorld.value()), "v");
        const Val l = Let(normalize(-light["direction"].xyz()), "l");
        const Val h = Let(normalize(v + l), "h");
        const Val nv = Let(max(dot(n, v), 1e-4), "nv");
        const Val nl = Let(max(dot(n, l), 0.0), "nl");
        const Val nh = Let(max(dot(n, h), 0.0), "nh");

        // dielectrics reflect 4%, metals tint reflection with their albedo
        const Val f0 = Let(mix(vec3(0.04), albedo, metallic), "f0");
        const Val f = Let(fresnelSchlick(max(dot(h, v), 0.0), f0), "f");
        const Val specular = Let(f * (distributionGGX(nh, roughness) *
                                      geometrySmith(nv, nl, roughness) / (4.0 * nv * nl + 1e-4)),
                                 "specular");
        const Val kd = Let((1.0 - f) * (1.0 - metallic), "kd");
        const Val diffuse = Let(kd * albedo / std::numbers::pi, "diffuse");

        const Val color = Var((diffuse + specular) * light["color"].rgb() * nl, "color");
        color.addAssign(albedo * 0.03);
        fragColor.set(vec4(color, material["baseColor"].a()));
      });

  sw.lap("build");
  if (!report(result)) return 1;
  printCode(result);
  sw.print(result.program->reflection.variantKey);
}
