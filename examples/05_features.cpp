// Newer DSL features in one program: single uniforms, struct and array
// values, Switch, While, a counting down For, compound assignment, matrix
// inverse, hyperbolic and packing functions, gradient and offset sampling.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <print>

using namespace moga;

int main() {
  Stopwatch sw;
  // single uniforms are packed into frameUniforms and materialUniforms
  const Val viewProj = uniform("viewProj", mat4f(), ResourceFrequency::Frame);
  const Val time = uniform("time", f32(), ResourceFrequency::Frame);
  const Val baseColor = uniform("baseColor", vec4f());
  const Texture albedo = texture("albedo", TextureDimension::D2);

  const TypePtr ring = makeStruct("Ring", {{"radius", f32()}, {"width", f32()}});

  const Val position = attribute("position", vec3f());
  const Val uv = attribute("uv", vec2f());
  const Varying vUv = varying("uv", vec2f());
  const Output color = output(0);

  const BuildResult result = buildShader(
      "features", variantKey("features", {}),
      [&] {
        vUv.set(uv);
        // inverse(inverse(m)) is m; it shows the WGSL helper at work
        Position(inverse(inverse(viewProj)) * vec4(position, 1.0));
      },
      [&] {
        const Val p = Let(vUv.value(), "p");
        const Val rings = Let(array({construct(ring, {0.15, 0.03}), construct(ring, {0.3, 0.02}),
                                     construct(ring, {0.45, 0.01})}),
                              "rings");
        const Val d = Let(length(p - 0.5), "d");
        const Val glow = Var(0.0, "glow");
        For({.start = 2, .end = 0, .step = -1, .condition = BinaryOp::Ge}, [&](const Val& i) {
          const Val r = Let(rings[i], "r");
          glow.addAssign(1.0 - smoothstep(0.0, r.member("width"), abs(d - r.member("radius"))));
        });

        // pick a tint by quadrant
        const Val tint = Var(vec3(1.0), "tint");
        const Val quadrant = Let(toI32(p.x() * 2.0) + toI32(p.y() * 2.0) * 2, "quadrant");
        Switch(quadrant)
            .Case(0, [&] { tint.set(vec3(1.0, 0.4, 0.3)); })
            .Case({1, 2}, [&] { tint.set(vec3(0.3, 0.8, 1.0)); })
            .Default([&] { tint.set(vec3(0.6, 1.0, 0.4)); });

        // halve the glow until it is below one
        const Val g = Var(glow * 4.0, "g");
        While(g > 1.0, [&] { g.divAssign(2.0); });

        const Val wave = Let(tanh(sin(time + d * 30.0) * 2.0) * 0.5 + 0.5, "wave");
        const Val texel = Let(albedo.sampleGrad(p, dpdx(p), dpdy(p)).rgb() * 0.5 +
                                  albedo.sampleOffset(p, ivec2(1, 0)).rgb() * 0.5,
                              "texel");
        // pack and unpack round trip through half floats
        const Val packed = Let(packHalf2x16(vec2(wave, g)), "packed");
        const Val back = Let(unpackHalf2x16(packed), "back");
        color.set(vec4(mix(texel * baseColor.rgb(), tint, back.y()) * back.x(), 1.0));
      });
  sw.lap("build");
  if (!report(result)) return 1;
  printCode(result);
  sw.print(result.program->reflection.variantKey);
}
