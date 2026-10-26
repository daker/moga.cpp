# 0004. One build entry point that returns structured diagnostics

- Status: Accepted
- Date: 2026-10-24
- Revised: 2026-10-26
- Topic: Build pipeline

## Context

Before this decision, a shader needed three calls: `program`,
`analyzeProgram`, then `compileGLSL` or `compileWGSL`. Each call had its own
result type.

Each mistake threw `ShaderError`. The error held a message and an optional
C++ location. Thus a new author saw only one error at a time. A tool had to
parse the text to find the build step, the shader stage or the resource.

The backends also stopped at the first missing feature. A program that
needed three WebGL extensions failed three times before it compiled.

Source locations were recorded only between `setShaderDebug(true)` and
`setShaderDebug(false)`. This thread local switch was easy to forget. It
stayed on when the code between the two calls threw.

The line maps (`LineMapping`) existed, but nothing used them to read a
driver error. Users had to find the C++ line by hand.

The library is in development, so the API can change.

## Decision

### 1. One entry point

- `buildShader(name, variantKey, vertex, fragment, options)` records the two
  stages, analyzes the program and compiles it.
- `buildComputeShader(name, variantKey, compute, body, options)` does the
  same for a compute program. It compiles WGSL only (ADR 0008).
- `BuildOptions` selects the targets:
  - `glsl` is a `std::optional<GLSLOptions>`. When it is empty, moga does not
    compile GLSL.
  - `wgsl` is a `bool`. When it is false, moga does not compile WGSL.
- `BuildResult` holds the program (reflection and line maps), the GLSL and
  WGSL output that the options ask for, and the diagnostics.
- `BuildResult` is `[[nodiscard]]`.

### 2. Diagnostics, not exceptions

- `buildShader` does not throw for a shader mistake.
- Each problem becomes a `Diagnostic` with these fields:
  - `severity`: error or warning.
  - `code`: a stable `ErrorCode`.
  - `step`: the build step (`Record`, `Analyze`, `TargetCheck`, `Emit` or
    `Driver`).
  - `stage`: the shader stage, when the problem is in one.
  - `where`: the C++ location, when it is known.
  - `generatedLine`: the line in the generated code, for a driver message.
  - `message`: the text, without the location.
- `BuildResult::ok()` is true when no diagnostic is an error.
- Exceptions are only for internal faults, for DSL calls with no open build,
  and for API misuse (for example `ShaderBuild::finish` called two times).

### 3. Error recovery while recording

- While a build is open, a guard wraps each public DSL operation.
- When an operation finds an error, the guard records a diagnostic. An
  expression then returns a placeholder (a poisoned `Val` of a plausible
  type). A statement does nothing.
- An expression made from a placeholder is also poisoned. moga does not
  report an error about a poisoned input, because the first error caused it.
  Thus moga reports each independent mistake one time.
- A declaration (type, resource, stage input or output, function) reports a
  failed check, but moga still makes it. Later code can use it.
- moga skips checks on a whole body (a missing return, a loop with no exit)
  when an error cut the body short.
- Analysis and emission do not run when recording had errors.
- With no open build, the DSL throws the first error.
- A new public DSL function must also use the guard. A test makes a list of
  mistakes outside a body and makes sure that no mistake escapes a build.

### 4. All target gaps at one time

- A target check step compares the features of the program with the
  capabilities of each target (ADR 0005, ADR 0008).
- It reports each missing extension or feature as one diagnostic with step
  `TargetCheck`.
- `unsupported(uses, capabilities)` does the comparison. It replaces the old
  `firstUnsupported()`.

### 5. Scoped source locations

- `BuildOptions::sourceLocations` replaces `setShaderDebug`.
- It is on by default when `NDEBUG` is not defined.
- The build sets and restores the recording state with a guard. Thus an
  early exit cannot leave it on.

### 6. Driver error mapping

- `BuildResult::mapGLSLLog(stage, log)` reads a WebGL info log.
- `BuildResult::mapWGSLLog(log)` reads Tint messages. WGSL is one module, so
  it takes no stage.
- Each returns diagnostics with code `Driver`, the generated line, and the C++
  line of the statement at or before that line.

### 7. Declarations before the stages

- Functions that the two stages share are usually made before the stages.
- `ShaderBuild` opens the build first. While it lives, errors in function
  bodies become diagnostics of that build, and statements record source
  locations.
- `ShaderBuild::finish()` records the stages and completes the build.
  `finishCompute()` does the same for a compute program.
- `buildShader()` is a `ShaderBuild` that finishes immediately.
- The declarations are plain locals that the stage lambdas capture. The
  lambdas have no parameters.
- Declarations take a defaulted `std::source_location`. Thus their errors
  show the line of the call.

### 8. One public way to build

- `program()`, `computeProgram()`, `analyzeProgram()`, `checkGLSL()`,
  `checkWGSL()`, `compileGLSL()` and `compileWGSL()` are in `src/Steps.hpp`.
- moga does not install this header.
- The tests and the benchmark include it to run one step at a time.

## Consequences

Positive:

- A new author writes one call and gets all the problems in the program.
  Each problem shows the C++ line.
- Tools (the viewer, a command line tool, an IDE) read fields. They do not
  parse messages.
- Target gaps show as one list that agrees with the options to set.
- There is no global debug state.

Negative:

- A guard wraps each public DSL function. A new function must also have
  one. A mistake in a function with no guard escapes the build.
- Callers must check `ok()`. When they do not, they get empty output, not a
  crash. Do not dereference `BuildResult::glsl` or `BuildResult::wgsl`
  before you check `ok()`.
- Source locations are on by default in Debug builds. This makes Debug
  builds slower. The `benchmark` example measures Release builds, so this
  does not change its result.

## Open work

These parts of the decision are not done in the code:

- `Diagnostic` has no field for the name of the resource or function, and no
  field for a suggested fix.
- Diagnostics from the `Analyze` and `Emit` steps have no `stage`, also when
  the stage is known (for example `WrongStage`). The cause is that
  `detail::Errors` (`src/Errors.hpp`) holds `ShaderError` values, which have
  no stage.
- Some analysis passes (bindings, layouts, names) and `src/Layout.cpp` still
  throw their first error. `buildShader` catches it, but the user sees only
  that first error.
- Many tests still check exceptions and message text. Tests must check the
  diagnostic code, not the message text.

## Related

- ADR 0005: validation architecture and backend capabilities.
- ADR 0008: WebGPU only features.
