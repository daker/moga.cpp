# 7. Diagnostics

`buildShader()` and `buildComputeShader()` do not throw for a shader
mistake. They return each problem as a `Diagnostic` in
`BuildResult::diagnostics`.

## Check the result

```cpp
const BuildResult r = buildShader("lit", key, vertex, fragment);
if (!r.ok()) {
  for (const Diagnostic& d : r.diagnostics) {
    if (d.where)
      std::println(stderr, "{}:{}: error[{}]: {}", d.where->file, d.where->line,
                   errorCodeName(d.code), d.message);
    else
      std::println(stderr, "error[{}]: {}", errorCodeName(d.code), d.message);
  }
  return;  // r.glsl and r.wgsl are empty
}
```

- `ok()` is true when no diagnostic is an error. Warnings do not change
  `ok()`.
- `BuildResult` is `[[nodiscard]]`.
- When the build fails, `glsl` and `wgsl` are empty. Do not dereference
  them before you check `ok()`.

`examples/common/Report.hpp` has a `printDiagnostics()` function that you
can copy.

## The fields of a diagnostic

| Field | Contents |
| --- | --- |
| `severity` | `Error` or `Warning`. |
| `code` | An `ErrorCode`. Use it in tests and tools. It does not change between versions. |
| `step` | The build step that found the problem: `Record`, `Analyze`, `TargetCheck`, `Emit` or `Driver`. |
| `stage` | The stage whose body was being recorded, when the problem is in one. |
| `where` | The C++ file, line and column, when they are known. |
| `generatedLine` | The line in the generated code, for a driver message. |
| `message` | The text, without the location. |

Note: at this time, only diagnostics from the `Record` step have a
`stage`. See [ADR 0004](../adr/0004-build-entry-point-and-diagnostics.md).

## One build reports all mistakes

While a build is open, a DSL operation that finds an error does not stop
the build:

- It records a diagnostic.
- An expression gives a placeholder value of a plausible type. Later code
  can use it.
- moga does not report a second error that the first error caused.
- A declaration with an error is still made.

Thus one build reports each independent mistake one time.

Analysis and emission do not run when recording had errors.

Note: some analysis checks (bindings, layouts and names) stop at their
first error. Correct it and build again to see the next one.

## Error codes

| Code | Meaning |
| --- | --- |
| `TypeMismatch` | A value has the wrong type for its location. |
| `InvalidOperands` | An operator, swizzle, member access or index does not apply to its operands. |
| `ArgumentCount` | A call or constructor has the wrong number of arguments. |
| `InvalidType` | A type that is not valid here (void, atomic, bool in memory, bad size). |
| `UnknownName` | A field, parameter or swizzle name that does not exist. |
| `DuplicateName` | Two declarations, fields or case values with the same name or value. |
| `ReservedName` | A name that moga keeps for itself. |
| `WrongStage` | Something used in a stage or function where it is not valid. |
| `NotAssignable` | A write to something that you cannot write. |
| `OutOfRange` | A constant, index or literal outside its range. |
| `ConstantRequired` | A value that must be a constant is not. |
| `InvalidConstant` | A constant calculation that is not defined (division by zero, overflow). |
| `ControlFlow` | Bad control flow (`Else` two times, `Break` outside a loop, a missing `Return`). |
| `StageInterface` | The stages do not fit together (varyings, outputs, `Position`). |
| `Layout` | A memory layout that is not valid, or that is different on the backends. |
| `TextureUsage` | A texture used in a way that its declaration does not permit. |
| `InvalidDeclaration` | Declaration options that are not valid together. |
| `Recursion` | A function calls itself. |
| `Uniformity` | A barrier or derivative that not all invocations reach together. |
| `Scope` | A local used outside the block that declares it. |
| `MissingFeature` | The target does not have a feature that the program uses. |
| `OutsideStage` | A statement outside a stage or function body. |
| `UniformWriter` | A `UniformWriter` call that does not agree with its layout. |
| `Driver` | A GPU driver rejected the generated code. |
| `Internal` | A fault in moga. Please report it. |

`errorCodeName(code)` gives the name as text.

## Target gaps

The target check step compares the program with each target. It reports
all missing features in one build. Each message names the option or the
extension to enable.

For example, a vertex stage that uses `drawIndex()` and `ClipDistance()`
gives two `MissingFeature` errors on WebGL2. When you set
`GLSLOptions::multiDraw` and `GLSLOptions::clipDistanceExtension`, the
errors go away.

## Driver errors

A driver can reject code that moga accepted, for example because of a
driver limit. Use the line maps to find the C++ line:

```cpp
BuildOptions options;
options.sourceLocations = true;
const BuildResult r = buildShader("lit", key, vertex, fragment, options);

// WebGL2: the info log of one stage
const std::vector<Diagnostic> v =
    r.mapGLSLLog(StageName::Fragment, gl.getShaderInfoLog(fragmentShader));

// WebGPU: the text of GPUCompilationInfo, or the browser console
const std::vector<Diagnostic> w = r.mapWGSLLog(compilationMessages);
```

- `mapGLSLLog` reads `ERROR: 0:line: ...` and `WARNING: ...` lines.
- `mapWGSLLog` reads `:line:column error|warning|note: ...` lines, as Tint
  writes them.
- Each result has code `Driver`, the `generatedLine`, and the C++ line of
  the statement at or before that line.
- Without source locations, the diagnostics have no C++ line.

## Errors with no open build

A DSL call outside a build throws `ShaderError` for an error. For example,
a declaration at namespace scope with a bad option throws.

`ShaderError` has `code()`, `message()` and `where()`. Its `what()` starts
with `file:line:column:` when the location is known.

`ShaderBuild::finish()` called two times throws `std::logic_error`. This is
API misuse, not a shader mistake.

See `examples/06_errors.cpp` for each case.
