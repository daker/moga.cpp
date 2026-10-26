# Architecture decision records

An ADR records one decision: its context, the decision and its consequences.
The ADRs are sorted by topic.

| ADR | Topic | Decision | Old number |
| --- | --- | --- | --- |
| [0001](0001-cpp23-cmake-conan.md) | Foundation | Build moga as a C++23 library with CMake and Conan | 0001 |
| [0002](0002-single-source-shader-dsl.md) | Foundation | Write shaders in a C++ DSL that compiles to GLSL ES 3.00 and WGSL | 0002 |
| [0003](0003-dsl-representation.md) | Foundation | Represent the DSL with variants, operators and source_location | 0004 |
| [0004](0004-build-entry-point-and-diagnostics.md) | Build pipeline | One build entry point that returns structured diagnostics | 0009 |
| [0005](0005-validation-architecture.md) | Build pipeline | Put checks in tables, one control flow analysis and backend capabilities | 0006 |
| [0006](0006-constant-values-as-ranges.md) | Build pipeline | Evaluate constant expressions as ranges of permitted values | 0008 |
| [0007](0007-uniform-and-storage-layout.md) | Data layout | One uniform layout, one storage layout and frequency based bind groups | 0003 |
| [0008](0008-webgpu-only-features.md) | Targets | Support WebGPU only features in WGSL and reject them in GLSL | 0005 |
| [0009](0009-optional-webgpu-device-features.md) | Targets | Support each optional WebGPU device feature that shaders can use | 0007 |

## Template

Each ADR has these parts:

- A header list: `Status`, `Date`, `Revised` (when changed), `Topic`.
- `Context`: the problem and the facts.
- `Decision`: what moga does.
- `Consequences`: the positive and the negative results.
- `Open work` (optional): parts of the decision that the code does not do
  yet.
- `Related`: links to other ADRs.

## Rules

- Do not change the decision of an accepted ADR. Write a new ADR that
  replaces it, and set the old status to `Superseded by ADR NNNN`.
- You can correct the text so that it agrees with the code. Add a
  `Revised` line.
- A new ADR gets the next free number. Put it in the table under its topic.
