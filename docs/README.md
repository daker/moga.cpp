# moga documentation

moga is a C++23 library. You write a shader one time, in C++. moga makes
GLSL ES 3.00 code for WebGL2 and WGSL code for WebGPU. It also gives the
reflection data that a renderer needs to bind the shader.

## User guide

Read the guide in this order. Each chapter uses the chapters before it.

| Chapter | Contents |
| --- | --- |
| [1. Getting started](guide/01-getting-started.md) | Install, build, and write the first shader. |
| [2. Concepts](guide/02-concepts.md) | Programs, stages, recording, build steps and variants. |
| [3. Types and values](guide/03-types-and-values.md) | Types, constructors, operators, variables and constants. |
| [4. Stage inputs, outputs and resources](guide/04-inputs-outputs-resources.md) | Attributes, varyings, outputs, builtins, uniforms, textures and tables. |
| [5. Control flow and functions](guide/05-control-flow-and-functions.md) | `If`, loops, `Switch` and user functions. |
| [6. Builtin functions](guide/06-builtin-functions.md) | Math, geometry, derivatives, bits and packing. |
| [7. Diagnostics](guide/07-diagnostics.md) | Errors, warnings, error codes and driver logs. |
| [8. Reflection and binding](guide/08-reflection-and-binding.md) | How a renderer binds a program on WebGL2 and WebGPU. |
| [9. Backends](guide/09-backends.md) | GLSL options, WGSL output and the differences between the backends. |
| [10. WebGPU only features](guide/10-webgpu-only.md) | Compute, storage, atomics, subgroups and device features. |

## Other documents

- [Architecture decision records](adr/README.md): why moga works as it does.
- [Examples](../examples/README.md): a tutorial in code.

## API reference

The public headers are in `include/moga/`. Include one header to get the full API:

```cpp
#include <moga/Shader.hpp>
```
