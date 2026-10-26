// The metallic roughness BRDF functions in the DSL: GGX
// distribution, correlated Smith visibility, Schlick Fresnel and Lambert
// diffuse, with optional anisotropy and clear coat as variants. GLSL `out`
// parameters become a returned struct (specular color and Fresnel term).
#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <numbers>
#include <optional>
#include <print>

using namespace moga;

namespace {

struct PbrFunctions {
  Function dGgx;
  Function vSmithCorrelated;
  Function fSchlick;
  Function diffuseLambert;
  Function specularIsotropic;
  std::optional<Function> specularAnisotropic;
};

/** The specular term and the Fresnel factor it used (the GLSL `out vec3 F`). */
TypePtr specularType() {
  static const TypePtr type = makeStruct("Specular", {{"color", vec3f()}, {"fresnel", vec3f()}});
  return type;
}

PbrFunctions buildFunctions(bool anisotropy) {
  const Val recPi = constant("RECIPROCAL_PI", 1.0 / std::numbers::pi);
  // each function is built after the ones it calls
  const Function dGgx = fn("D_GGX", {{"NdH", f32()}, {"roughness", f32()}}, f32(), [](const Args& a) {
    const Val alpha = Let(a["roughness"] * a["roughness"], "a");
    const Val a2 = Let(alpha * alpha, "a2");
    const Val d = Let((a["NdH"] * a2 - a["NdH"]) * a["NdH"] + 1.0, "d");
    const Val denom = Let(max(consts::pi() * d * d, 1e-8), "denom");
    Return(a2 / denom);
  });
  const Function vSmithCorrelated =
      fn("V_SmithCorrelated", {{"NdV", f32()}, {"NdL", f32()}, {"roughness", f32()}}, f32(), [](const Args& a) {
        const Val a2 = Let(a["roughness"] * a["roughness"], "a2");
        const Val ggxV = Let(a["NdL"] * sqrt(a2 + a["NdV"] * (a["NdV"] - a2 * a["NdV"])), "ggxV");
        const Val ggxL = Let(a["NdV"] * sqrt(a2 + a["NdL"] * (a["NdL"] - a2 * a["NdL"])), "ggxL");
        Return(0.5 / (ggxV + ggxL));
      });
  const Function fSchlick =
      fn("F_Schlick", {{"F0", vec3f()}, {"F90", vec3f()}, {"HdL", f32()}}, vec3f(), [](const Args& a) {
        Return(a["F0"] + (a["F90"] - a["F0"]) * pow(oneMinus(a["HdL"]), 5.0));
      });
  const Function diffuseLambert =
      fn("DiffuseLambert", {{"albedo", vec3f()}}, vec3f(), [&](const Args& a) { Return(a["albedo"] * recPi); });
  const Function specularIsotropic =
      fn("SpecularIsotropic",
         {{"NdH", f32()}, {"NdV", f32()}, {"NdL", f32()}, {"HdL", f32()}, {"roughness", f32()},
          {"F0", vec3f()}, {"F90", vec3f()}},
         specularType(), [&](const Args& a) {
           const Val d = Let(dGgx(a["NdH"], a["roughness"]), "D");
           const Val v = Let(vSmithCorrelated(a["NdV"], a["NdL"], a["roughness"]), "V");
           const Val fresnel = Let(fSchlick(a["F0"], a["F90"], a["HdL"]), "F");
           Return(construct(specularType(), {(d * v) * fresnel, fresnel}));
         });
  PbrFunctions f{dGgx, vSmithCorrelated, fSchlick, diffuseLambert, specularIsotropic, std::nullopt};
  if (!anisotropy) return f;

  const Function dAniso = fn("D_GGX_Anisotropic",
                             {{"at", f32()}, {"ab", f32()}, {"TdH", f32()}, {"BdH", f32()}, {"NdH", f32()}},
                             f32(), [&](const Args& a) {
                               const Val a2 = Let(a["at"] * a["ab"], "a2");
                               const Val d = Let(vec3(a["ab"] * a["TdH"], a["at"] * a["BdH"], a2 * a["NdH"]), "d");
                               const Val d2 = Let(dot(d, d), "d2");
                               const Val b2 = Let(a2 / d2, "b2");
                               Return(a2 * b2 * b2 * recPi);
                             });
  const Function vAniso = fn("V_SmithGGXCorrelated_Anisotropic",
                             {{"at", f32()}, {"ab", f32()}, {"TdV", f32()}, {"BdV", f32()}, {"TdL", f32()},
                              {"BdL", f32()}, {"NdV", f32()}, {"NdL", f32()}},
                             f32(), [](const Args& a) {
                               const Val lambdaV =
                                   Let(a["NdL"] * length(vec3(a["at"] * a["TdV"], a["ab"] * a["BdV"], a["NdV"])),
                                       "lambdaV");
                               const Val lambdaL =
                                   Let(a["NdV"] * length(vec3(a["at"] * a["TdL"], a["ab"] * a["BdL"], a["NdL"])),
                                       "lambdaL");
                               Return(0.5 / (lambdaV + lambdaL));
                             });
  f.specularAnisotropic =
      fn("SpecularAnisotropic",
         {{"at", f32()}, {"ab", f32()}, {"l", vec3f()}, {"t", vec3f()}, {"b", vec3f()}, {"h", vec3f()},
          {"TdV", f32()}, {"BdV", f32()}, {"NdH", f32()}, {"NdV", f32()}, {"NdL", f32()}, {"HdL", f32()},
          {"F0", vec3f()}, {"F90", vec3f()}},
         specularType(), [&](const Args& a) {
           const Val tdl = Let(dot(a["t"], a["l"]), "TdL");
           const Val bdl = Let(dot(a["b"], a["l"]), "BdL");
           const Val tdh = Let(dot(a["t"], a["h"]), "TdH");
           const Val bdh = Let(dot(a["b"], a["h"]), "BdH");
           // specular anisotropic BRDF
           const Val d = Let(dAniso(a["at"], a["ab"], tdh, bdh, a["NdH"]), "D");
           const Val v = Let(vAniso(a["at"], a["ab"], a["TdV"], a["BdV"], tdl, bdl, a["NdV"], a["NdL"]), "V");
           const Val fresnel = Let(f.fSchlick(a["F0"], a["F90"], a["HdL"]), "F");
           Return(construct(specularType(), {(d * v) * fresnel, fresnel}));
         });
  return f;
}

BuildResult buildPbr(bool anisotropy, bool clearCoat) {
  // opened first, so an error in the functions below becomes a diagnostic of this build
  const VariantOptions options{{"anisotropy", anisotropy}, {"clearCoat", clearCoat}};
  ShaderBuild build("pbrfns", variantKey("pbrfns", options));
  const UniformBlock camera =
      uniformBlock("camera", {{"viewProj", mat4f()}, {"eye", vec4f()}}, ResourceFrequency::Frame);
  const Val metallicUniform = uniform("metallicUniform", f32());
  const Val roughnessUniform = uniform("roughnessUniform", f32());
  const Val emissiveFactorUniform = uniform("emissiveFactorUniform", vec3f());
  const Val aoStrengthUniform = uniform("aoStrengthUniform", f32());
  const Val baseF0Uniform = uniform("baseF0Uniform", f32());
  const Val edgeTintUniform = uniform("edgeTintUniform", vec3f());
  const Val baseColor = uniform("baseColorUniform", vec3f());

  const Val position = attribute("position", vec3f());
  const Val normal = attribute("normal", vec3f());
  const Val tangent = attribute("tangent", vec3f());
  const Varying vNormal = varying("normal", vec3f());
  const Varying vTangent = varying("tangent", vec3f());
  const Varying vWorld = varying("world", vec3f());
  const Output fragColor = output(0);

  const PbrFunctions f = buildFunctions(anisotropy);
  return build.finish(
      [&] {
        vNormal.set(normal);
        vTangent.set(tangent);
        vWorld.set(position);
        Position(camera["viewProj"] * vec4(position, 1.0));
      },
      [&] {
        const Val n = Let(normalize(vNormal.value()), "n");
        const Val v = Let(normalize(camera["eye"].xyz() - vWorld.value()), "v");
        const Val l = Let(normalize(vec3(0.3, 1.0, 0.5)), "l");
        const Val h = Let(normalize(l + v), "h");
        // the 1e-4 floor keeps grazing angles away from a division by zero
        const Val NdV = Let(max(dot(n, v), 1e-4), "NdV");
        const Val NdL = Let(max(dot(n, l), 0.0), "NdL");
        const Val NdH = Let(max(dot(n, h), 0.0), "NdH");
        const Val HdL = Let(max(dot(h, l), 0.0), "HdL");
        const Val roughness = Let(clamp(roughnessUniform, 0.045, 1.0), "roughness");
        const Val metallic = Let(metallicUniform, "metallic");
        const Val F0 = Let(mix(vec3(baseF0Uniform), baseColor, metallic), "F0");
        const Val F90 = Let(mix(vec3(1.0), edgeTintUniform, metallic), "F90");

        const Val specular = Var(specularType(), "specular");
        if (anisotropy) {
          const Val anisotropyUniform = uniform("anisotropyUniform", f32());
          // Gram Schmidt: a tangent at right angles to the normal, and the bitangent
          const Val t = Let(normalize(vTangent.value() - n * dot(n, vTangent.value())), "t");
          const Val b = Let(cross(n, t), "b");
          const Val at = Let(max(roughness * (1.0 + anisotropyUniform), 0.001), "at");
          const Val ab = Let(max(roughness * oneMinus(anisotropyUniform), 0.001), "ab");
          specular.set((*f.specularAnisotropic)(at, ab, l, t, b, h, dot(t, v), dot(b, v), NdH, NdV, NdL, HdL, F0, F90));
        } else {
          specular.set(f.specularIsotropic(NdH, NdV, NdL, HdL, roughness, F0, F90));
        }
        const Val diffuse =
            Let(f.diffuseLambert(baseColor) * oneMinus(metallic) * (vec3(1.0) - specular.member("fresnel")), "diffuse");
        const Val color = Var((diffuse + specular.member("color")) * NdL, "color");

        if (clearCoat) {
          const Val coatF0Uniform = uniform("coatF0Uniform", f32());
          const Val coatStrengthUniform = uniform("coatStrengthUniform", f32());
          const Val coatRoughnessUniform = uniform("coatRoughnessUniform", f32());
          const Val coatColorUniform = uniform("coatColorUniform", vec3f());
          const Val coat = Let(f.specularIsotropic(NdH, NdV, NdL, HdL, clamp(coatRoughnessUniform, 0.045, 1.0),
                                                   vec3(coatF0Uniform), vec3(1.0)),
                               "coat");
          const Val coatFresnel = Let(coat.member("fresnel") * coatStrengthUniform, "coatFresnel");
          // the coat reflects part of the light; the layer below gets the rest, tinted by the coat
          color.set(color * (vec3(1.0) - coatFresnel) * coatColorUniform +
                    coat.member("color") * coatStrengthUniform * NdL);
        }
        const Val ambient = Let(baseColor * 0.03 * aoStrengthUniform, "ambient");
        fragColor.set(vec4(color + ambient + emissiveFactorUniform, 1.0));
      });
}

} // namespace

int main() {
  for (const bool anisotropy : {false, true}) {
    for (const bool clearCoat : {false, true}) {
      Stopwatch sw;
      const BuildResult result = buildPbr(anisotropy, clearCoat);
      sw.lap("build");
      if (!report(result)) return 1;
      std::println("// ==== {}", result.program->reflection.variantKey);
      printCode(result);
      sw.print(result.program->reflection.variantKey);
    }
  }
}
