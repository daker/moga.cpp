# moga.cpp

C++23 shader DSL: write a shader once in C++, compile it
to GLSL ES 3.00 (WebGL2) and WGSL (WebGPU) with reflection and a shared
std140 / WGSL uniform layout.

```cpp
const Val position = attribute("position", vec3f());
const Varying vColor = varying("color", vec3f());
const Output color = output(0);
const BuildResult r = buildShader("tri", "TRI",
    [&] { vColor.set(position); Position(vec4(position, 1.0)); },
    [&] { color.set(vec4(vColor.value(), 1.0)); });
if (!r.ok()) {
  for (const Diagnostic& d : r.diagnostics) std::println("{}", d.message);
  return;  // glsl and wgsl are empty when the build failed
}
std::string wgsl = r.wgsl->code;
std::string vertex = r.glsl->vertex;
```

`buildShader()` never throws for a shader mistake: every problem it finds
becomes a `Diagnostic` with a code, the build step, the stage and the C++
line (ADR 0004). `buildComputeShader()` does the same for compute programs.
To declare functions shared by both stages first, open a `ShaderBuild`:

```cpp
ShaderBuild build("lit", "LIT");
const Function shade = fn("shade", {{"n", vec3f()}}, f32(), [](const Args& a) { ... });
const BuildResult r = build.finish([&] { ... }, [&] { ... shade(n) ... });
```

## Build

```sh
conan install . -s compiler.cppstd=23 --build=missing
cmake --preset conan-default
cmake --build --preset conan-release
ctest --preset conan-release
```

## Examples

The examples in `examples/` are a tutorial, from `01_triangle.cpp` to
`10_compute.cpp`, plus `11_benchmark.cpp`. `examples/README.md` lists what
each one shows.

## Documentation

- The user guide starts at `docs/README.md`: getting started, concepts,
  the DSL, diagnostics, reflection and binding, backends and WebGPU only
  features.
- Optional WebGPU device features that shaders use (f16, dual source
  blending, primitive index, subgroup size, storage texture format tiers,
  bgra8unorm storage, clip distances, float32 filtering) are listed in
  `Reflection::deviceFeatures`, ready for `requiredFeatures` (ADR 0009).
- Decisions are recorded in `docs/adr/`. `docs/adr/README.md` is the index.

## Viewing the output in a browser

Open `examples/web/viewer.html` in Chrome, paste the full
output of an example and press Render. It splits the output into GLSL vertex,
GLSL fragment and WGSL, shows compile errors, and draws a lit sphere with
WebGL2 and WebGPU side by side. Uniforms, textures and tables are filled from
the reflected layout using field names (`viewProj`, `eye`, `baseColor`, ...);
the values used are listed under "Uniform values used".
