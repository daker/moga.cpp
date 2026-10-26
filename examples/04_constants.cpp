// Named constants: the built in PI, TWO_PI, HALF_PI, EPSILON and INFINITY,
// plus user constants built from other constants. Each program emits only
// the constants it uses, after the constants they depend on.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <print>

using namespace moga;

int main() {
  // Declared one time, shared by any number of programs.
  const Val goldenAngle = constant("GOLDEN_ANGLE", consts::pi() * (3.0 - sqrt(5.0)));
  const Val tint = constant("TINT", vec3(1.0, 0.6, 0.2));

  const Val position = attribute("position", vec3f());
  const Varying vPos = varying("pos", vec2f());
  const Output color = output(0);

  Stopwatch sw;
  const BuildResult result = buildShader(
      "constants", variantKey("constants", {}),
      [&] {
        vPos.set(position.xy());
        Position(vec4(position, 1.0));
      },
      [&] {
        const Val p = Let(vPos.value(), "p");
        // polar angle in 0..1
        const Val angle = Let((atan2(p.y(), p.x()) + consts::pi()) / consts::twoPi(), "angle");
        const Val radius = Let(length(p), "radius");
        // a nearest distance search starts at INFINITY
        const Val nearest = Var(consts::infinity(), "nearest");
        For(0, 32, [&](const Val& i) {
          const Val a = Let(toF32(i) * goldenAngle, "a");
          const Val r = Let(sqrt(toF32(i) / 32.0) * 0.6, "r");
          const Val seed = Let(vec2(cos(a), sin(a)) * r, "seed");
          nearest.set(min(nearest, distance(p, seed)));
        });
        const Val rings = Let(0.5 + 0.5 * sin(radius * 40.0 - consts::halfPi()), "rings");
        const Val dots = Let(1.0 - smoothstep(0.02, 0.03, nearest), "dots");
        // EPSILON keeps the division finite at the center
        const Val shade = Let(angle * rings / max(radius, consts::epsilon()), "shade");
        color.set(vec4(mix(tint * min(shade, 1.0), vec3(1.0), dots), 1.0));
      });
  sw.lap("build");
  if (!report(result)) return 1;
  printCode(result);
  sw.print(result.program->reflection.variantKey);
}
