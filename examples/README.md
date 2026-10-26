# Examples

The examples are a tutorial. Read them in number order. Each example adds
new features to the ones before it.

| File | Target | What it shows |
| --- | --- | --- |
| `01_triangle.cpp` | `example_triangle` | The smallest program: a color for each vertex, printed for the two backends. |
| `02_phong.cpp` | `example_phong` | Uniform blocks, a texture, a helper function and a variant. |
| `03_instanced.cpp` | `example_instanced` | Tables (a record for each instance), a loop, reflection and the uniform writer. |
| `04_constants.cpp` | `example_constants` | Built in and user named constants. |
| `05_features.cpp` | `example_features` | Single uniforms, structs, arrays, `Switch`, `While`, `inverse`, packing, gradient and offset sampling. |
| `06_errors.cpp` | `example_errors` | Diagnostics: error codes, build steps, stages, C++ lines and driver log mapping. |
| `07_clipping.cpp` | `example_clipping` | Clip distances, or a fallback that discards fragments. |
| `08_pbr.cpp` | `example_pbr` | Metallic roughness PBR with GGX. |
| `09_pbrfns.cpp` | `example_pbrfns` | The BRDF functions with anisotropy and clear coat variants. |
| `10_compute.cpp` | `example_compute` | WebGPU only: compute, storage, atomics, workgroup memory and subgroups. |
| `11_benchmark.cpp` | `example_benchmark` | The generation time of each build step, for programs of 1 to 200 functions. Build it in Release. |

Other files:

- `common/Report.hpp` prints the diagnostics and the generated code.
- `common/Stopwatch.hpp` prints the time of each step.
- `web/viewer.html` shows the output of an example in a browser.

## Build and run

The examples build when `MOGA_BUILD_EXAMPLES` is on (the default):

```sh
cmake --build --preset conan-release --target example_triangle
```

## See the output in a browser

1. Open `web/viewer.html` in Chrome. You do not need a server.
2. Paste the full output of an example.
3. Push Render.

The viewer splits the output into GLSL vertex, GLSL fragment and WGSL. It
shows compile errors. It draws a lit sphere with WebGL2 and WebGPU side by
side.
