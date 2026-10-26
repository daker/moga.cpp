// Clipping against up to four user planes (ax + by + cz + d >= 0 keeps a
// point). Two variants: hardware clip distances written by the vertex stage,
// and a fallback that discards fragments for devices without them.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <print>

using namespace moga;

namespace {

constexpr int kPlanes = 4;

BuildResult buildClipping(bool clipDistances) {
  const UniformBlock camera =
      uniformBlock("camera", {{"viewProj", mat4f()}}, ResourceFrequency::Frame);
  // planes in world space; count is how many are active
  const UniformBlock clip = uniformBlock(
      "clip", {{"planes", arrayOf(vec4f(), kPlanes)}, {"count", i32()}}, ResourceFrequency::Object);

  const Val position = attribute("position", vec3f());
  const Varying vWorld = varying("world", vec3f());
  const Output fragColor = output(0);

  /** Signed distance of `p` to plane `i`, or 1 when the plane is inactive. */
  const Function planeDistance =
      fn("planeDistance", {{"p", vec3f()}, {"i", i32()}}, f32(), [&](const Args& a) {
        If(a["i"] >= clip["count"], [] { Return(1.0); });
        Return(dot(clip["planes"][a["i"]], vec4(a["p"], 1.0)));
      });

  const VariantOptions options{{"clipDistances", clipDistances}};
  return buildShader(
      "clipping", variantKey("clipping", options),
      [&] {
        const Val world = Let(vec4(position, 1.0), "world");
        vWorld.set(world.xyz());
        // clip distance indices must be constants, so the loop is unrolled here
        if (clipDistances)
          for (int i = 0; i < kPlanes; ++i) ClipDistance(i, planeDistance(world.xyz(), i));
        Position(camera["viewProj"] * world);
      },
      [&] {
        if (!clipDistances) {
          For(0, clip["count"], [&](const Val& i) {
            If(planeDistance(vWorld.value(), i) < 0.0, [] { Discard(); });
          });
        }
        fragColor.set(vec4(0.8, 0.8, 0.8, 1.0));
      },
      {.glsl = GLSLOptions{.clipDistanceExtension = true}});
}

} // namespace

int main() {
  for (const bool clipDistances : {true, false}) {
    Stopwatch sw;
    const BuildResult result = buildClipping(clipDistances);
    sw.lap("build");
    if (!report(result)) return 1;
    std::println("// ==== {} ({} clip distances)", result.program->reflection.variantKey,
                 result.program->reflection.clipDistances);
    printCode(result);
    sw.print(result.program->reflection.variantKey);
  }
}
