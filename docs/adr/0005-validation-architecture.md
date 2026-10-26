# 0005. Put checks in tables, one control flow analysis and backend capabilities

- Status: Accepted
- Date: 2026-10-23
- Revised: 2026-10-26
- Topic: Build pipeline

## Context

Audits added approximately 190 checks. Each check went where its bug was
found. Thus one concept had rules in many places:

- The argument rules of a builtin were in its DSL function.
- Its backend names were in two switches.
- Its use in constants was in `constEvaluable()`.
- Its helper needs were in `recordHelper()`.

Five functions walked the statements to answer control flow questions. Each
did this in a different way. Most recent bugs came from the differences
between them.

Backend gaps were special cases in the analysis and in the GLSL compiler.

The backends changed the code while they printed it.

Several functions were at the complexity limit of 20.

## Decision

The work has four steps. After each step, the test suite must pass with no
changes.

### 1. One control flow analysis

- Each statement has a set of behaviors, as in WGSL. A behavior is a way to
  leave the statement: next, break, continue or return.
- Fixed rules combine the behaviors of blocks, branches, switches and loops.
- An optional marker statement ends a path as satisfied. Thus the same
  analysis finds out if each path writes `Position()`.
- Loop exits, function returns and position coverage are queries on this
  analysis.

### 2. A builtin table

- The table (`src/BuiltinTable.cpp`) has one row for each builtin.
- A row holds the overloads as type patterns, the backend names or
  lowerings, the compile time evaluation and the stages where the builtin is
  allowed.
- A row uses the type domain of the first argument and a result rule as its
  key.
- One overload resolver replaces the checks in each function.
- The shape rules for the other arguments (broadcast, same types) stay in
  the DSL functions.

### 3. Backend capabilities

- The analysis records the features that a program uses. They are in
  `Reflection::features` as `FeatureUse{feature, what}` entries, first use
  first.
- Each backend declares what it supports: `webgpuCapabilities()` and
  `webgl2Capabilities(firstVertexConvention, blendFuncExtended)`.
- `unsupported(uses, capabilities)` does one comparison. It returns each use
  that the backend does not support.
- These are in `include/moga/Capabilities.hpp`.

### 4. Separate passes

- **Build**: makes the AST and checks types.
- **Validate**: checks the program rules.
- **Lower**: makes the backend specific changes: evaluate one time,
  diagonals, swizzle writes, `mod`.
- **Print**: writes the text. It makes no decisions.

For expressions, `lower(target, expr)` in `src/Lower.cpp` selects a form
(native, inline, helper or operand) and the helper to declare. The analysis
collects `glslHelpers` and `wgslHelpers` from it. The printers write the
selected form. They do not test for side effects.

The lowering is a decision for each node. It does not change the AST. Thus
the output text stays the same.

WGSL swizzle writes and the empty switch default stay in the printer. They
are syntax, with no decision to make.

### Later additions

Two later additions use the same passes:

- **Uniformity** (`src/Uniformity.cpp`) is a program rule. Thus it runs in
  the validate pass.
  - All invocations must reach barriers and derivatives together.
  - moga tracks values and control flow as WGSL does.
  - There is one simplification, on the strict side: a local that gets an
    uneven value at one location is non uniform at all locations.
  - Programs with no barriers and no derivatives skip this check.
- **Shared values.** A `Val` that the code uses again is a graph, not a
  tree. Each pass works on the nodes, not on the expanded tree.
  - `makeExpr()` sets the facts that the passes need one time: side
    effects, constant leaves, uniformity relevance and expanded size.
  - Recursive checks keep their results (memoization).
  - When a large statement uses a pure value again, the printers write that
    value one time, as a local.

## Consequences

Positive:

- A rule is in one location. To add a builtin or a backend, you add data.
- Control flow checks cannot disagree, because they use one analysis.
- Each lowering is a small function with its own tests.

Negative:

- The refactor was large. It was done in four steps, and the existing tests
  were the safety net.

## Open work

The code does not fully agree with steps 3 and 4:

- The printers still make some checks that an earlier pass already makes:
  - `src/WGSL.cpp`: 3D depth textures, bitcast constants and `DrawIndex`.
  - `src/GLSL.cpp`: atomic types.
  - `src/TextEmitter.cpp`: non finite literals.
  These checks must move to the validate pass, or become `Internal` errors.
- Some target checks are not in the capability tables: clip distances and
  `DrawIndex` in `src/GLSL.cpp`, and `DrawIndex` and `LocalInstanceIndex` in
  `src/WGSL.cpp`. They must become `Feature` values.
- The printers select some helpers by the builtin (`Smoothstep`,
  `Saturate`, `Inverse`), not from the lowering. The lowering must give the
  helper.
- `constEvaluable` rules in `src/Known.cpp` still switch on the builtin.
  The table must hold them.

## Related

- ADR 0004: build entry point and diagnostics (target check step).
- ADR 0006: constant values as ranges.
- ADR 0008: WebGPU only features.
