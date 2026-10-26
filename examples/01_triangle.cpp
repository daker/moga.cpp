// Smallest program: per vertex color, printed for both backends.

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <print>

using namespace moga;

int main() {
  Stopwatch sw;
  const Val position = attribute("position", vec3f());
  const Val color = attribute("color", vec3f());
  const Varying vColor = varying("color", vec3f());
  const Output fragColor = output(0);

  const BuildResult result = buildShader(
      "triangle", variantKey("triangle", {}),
      [&] {
        vColor.set(color);
        Position(vec4(position, 1.0));
      },
      [&] { fragColor.set(vec4(vColor.value(), 1.0)); });

  sw.lap("build");
  if (!report(result)) return 1;
  printCode(result);
  sw.print(result.program->reflection.variantKey);
}
