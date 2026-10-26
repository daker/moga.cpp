# 1. Getting started

## Requirements

- A C++23 compiler with a standard library that has `std::print`.
- CMake 3.25 or later.
- Conan 2.

The library has no runtime dependencies. The tests use doctest.

## Build moga

Run these commands in the root of the repository:

```sh
conan install . -s compiler.cppstd=23 --build=missing
cmake --preset conan-default
cmake --build --preset conan-release
ctest --preset conan-release
```

Two CMake options control the optional targets:

| Option | Default | Effect |
| --- | --- | --- |
| `MOGA_BUILD_EXAMPLES` | `ON` | Builds the programs in `examples/`. |
| `MOGA_BUILD_TESTS` | `ON` | Builds the test suite. |

## Use moga in your project

After `conan create .` or `cmake --install`, find the package and link the
target:

```cmake
find_package(moga CONFIG REQUIRED)
target_link_libraries(my_renderer PRIVATE moga::moga)
```

## Write the first shader

This program gives each vertex a color. The fragment stage writes the
interpolated color.

```cpp
#include <moga/Shader.hpp>

#include <print>

using namespace moga;

int main() {
  // 1. Declare the inputs and outputs.
  const Val position = attribute("position", vec3f());
  const Val color = attribute("color", vec3f());
  const Varying vColor = varying("color", vec3f());
  const Output fragColor = output(0);

  // 2. Build the program. The two lambdas are the vertex and fragment stages.
  const BuildResult result = buildShader(
      "triangle", variantKey("triangle", {}),
      [&] {
        vColor.set(color);
        Position(vec4(position, 1.0));
      },
      [&] { fragColor.set(vec4(vColor.value(), 1.0)); });

  // 3. Check for errors before you use the output.
  if (!result.ok()) {
    for (const Diagnostic& d : result.diagnostics) std::println("{}", d.message);
    return 1;
  }

  // 4. Use the code.
  std::println("{}", result.glsl->vertex);
  std::println("{}", result.glsl->fragment);
  std::println("{}", result.wgsl->code);
}
```

Step by step:

1. `attribute()`, `varying()` and `output()` declare the interface of the
   program. Declare them before the build.
2. `buildShader()` runs the vertex lambda, then the fragment lambda. Each
   DSL call in a lambda records a statement. The C++ code does not calculate
   values. It records shader code.
3. `buildShader()` does not throw for a shader mistake. It returns
   diagnostics. Always check `ok()`.
4. `result.glsl` and `result.wgsl` hold the generated code. They are empty
   when the build failed.

## See the output

The examples print their output in a format that the viewer reads:

1. Build and run an example, for example `example_triangle`.
2. Open `examples/web/viewer.html` in Chrome.
3. Paste the full output, then push Render.

The viewer draws the shader with WebGL2 and WebGPU side by side.

## Next steps

- Read [2. Concepts](02-concepts.md).
- Read the examples in number order. `examples/README.md` lists them.
