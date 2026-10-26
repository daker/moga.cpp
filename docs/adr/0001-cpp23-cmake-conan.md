# 0001. Build moga as a C++23 library with CMake and Conan

- Status: Accepted
- Date: 2026-10-21
- Revised: 2026-10-26
- Topic: Foundation

## Context

moga makes shaders for renderers on Windows, Linux and macOS. Existing
native projects must be able to use it with no special work.

The implementation uses modern library features: `std::format`,
`std::variant`, concepts and `std::source_location`.

## Decision

- moga is a static C++23 library. The CMake target is `moga::moga`.
- CMake 3.25 or later is the build system. The library has install and
  export rules.
- The examples and the tests are optional targets. The options
  `MOGA_BUILD_EXAMPLES` and `MOGA_BUILD_TESTS` control them.
- Conan 2 manages the dependencies and makes the package.
- The library has no runtime dependencies. doctest is a test dependency only.
- The library reports shader mistakes as diagnostics (ADR 0004).

## Consequences

Positive:

- Users consume moga with standard CMake and Conan steps.
- No third party code goes into the user's program.
- Modern library features keep the code small.

Negative:

- Users must have a C++23 compiler.

## Related

- ADR 0004: build entry point and diagnostics.
