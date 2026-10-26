# 2. Concepts

## The program and its stages

A **program** is a set of shader stages that run together.

- A **render program** has a vertex stage and a fragment stage.
- A **compute program** has one compute stage. It is WebGPU only. See
  [10. WebGPU only features](10-webgpu-only.md).

Each stage is a C++ lambda. moga calls the lambda one time, when you build
the program.

## Recording, not running

DSL calls do not calculate values. They record shader code.

```cpp
const Val n = Let(normalize(vNormal.value()), "n");
```

This line does not normalize a vector in C++. It adds a statement to the
program. The statement declares a constant `n` that holds the normalized
varying.

Thus:

- C++ `if` and `for` run when the program is built. Use them to make
  variants, for example to add a texture sample only when a texture exists.
- DSL `If`, `For` and `While` become shader control flow. They run on the
  GPU.

```cpp
if (textured) base.mulAssign(albedo.sample(uv).rgb());  // C++: decides the variant
If(frontFacing(), [&] { color.addAssign(specular); });   // DSL: runs on the GPU
```

## Values

`Val` is a handle to a shader expression. It is immutable. Operators on
`Val` make new expressions:

```cpp
const Val world = camera["viewProj"] * vec4(position, 1.0);
```

A C++ number in an expression becomes a literal. The literal gets the type
of the other operand. Thus `v * 0.5` is valid for an `f32` vector, and
`i + 1` is valid for an `i32` value.

A `Val` that you use two times is one node, not two copies. moga writes a
large shared value one time, as a local.

See [3. Types and values](03-types-and-values.md).

## Declarations

Declarations make the interface of a program:

- stage inputs and outputs: `attribute()`, `varying()`, `output()`
- resources: `uniform()`, `uniformBlock()`, `texture()`, `table()`,
  `storage()`, `storageTexture()`
- functions: `fn()`
- named constants: `constant()`

Make declarations before the stages. Many programs can use the same
declaration. A declaration is in a program only when a stage uses it, or
when you retain it (see [8. Reflection and binding](08-reflection-and-binding.md)).

## The build

`buildShader()` does all the steps in one call:

| Step | `BuildStep` | What it does |
| --- | --- | --- |
| Record | `Record` | Runs the stage lambdas and checks each operation. |
| Analyze | `Analyze` | Finds what each stage uses, checks the program rules and assigns bindings. |
| Target check | `TargetCheck` | Compares the features of the program with each target. |
| Emit | `Emit` | Writes GLSL and WGSL. |

`BuildOptions` selects the targets:

```cpp
BuildOptions options;
options.glsl = GLSLOptions{.clipControl = true};  // empty: no GLSL
options.wgsl = true;                               // false: no WGSL
const BuildResult r = buildShader("lit", key, vertex, fragment, options);
```

`BuildResult` holds:

| Field | Contents |
| --- | --- |
| `program` | The analyzed program. `program->reflection` is the reflection. |
| `glsl` | The GLSL vertex and fragment code, and their line maps. |
| `wgsl` | The WGSL module, its entry point names and its line map. |
| `diagnostics` | All errors and warnings. |

When the build fails, the fields that a failed step did not make are empty.
Always check `ok()` first.

## Functions shared by the two stages

Use `ShaderBuild` when you declare functions before the stages. While a
`ShaderBuild` lives, errors in function bodies become diagnostics of that
build:

```cpp
ShaderBuild build("lit", key);
const Function shade = fn("shade", {{"n", vec3f()}}, f32(), [](const Args& a) {
  Return(max(dot(a["n"], vec3(0.0, 1.0, 0.0)), 0.0));
});
const BuildResult r = build.finish(
    [&] { /* vertex */ },
    [&] { /* fragment: shade(n) */ });
```

`buildShader()` is a `ShaderBuild` that finishes immediately.

With no open build, a DSL call that finds an error throws `ShaderError`.

## Variants

A **variant** is one version of a program, for a set of options.
`variantKey()` makes a key from a base name and the options:

```cpp
const VariantOptions options{{"tex", true}, {"lights", 4.0}, {"mode", std::string("pbr")}};
const std::string key = variantKey("lit", options);
// lit|lights=4|mode="pbr"|tex=true
```

- The options are sorted by name. Their order does not change the key.
- Two different option sets never give the same key.
- Use the key as a cache key for the compiled program on the two backends.

## Threads

The build state is thread local. You can build different programs on
different threads at the same time. Use one `ShaderBuild` for each thread at
a time.

## Source locations

`BuildOptions::sourceLocations` records the C++ line of each statement.

- It is on by default when `NDEBUG` is not defined.
- With it, diagnostics found after recording show the C++ line.
- With it, the line maps connect generated lines to C++ lines.

See [7. Diagnostics](07-diagnostics.md).
