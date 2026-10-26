# 0003. Represent the DSL with variants, operators and source_location

- Status: Accepted
- Date: 2026-10-21
- Revised: 2026-10-26
- Topic: Foundation

## Context

The DSL must look like shader code, but it must stay plain C++.

It must be easy to add new node kinds.

Programs must be buildable from more than one thread.

## Decision

- AST nodes are `std::variant` payloads (`ex::*` for expressions, `st::*` for
  statements). An `Overloaded` helper visits them.
- Expressions are immutable `shared_ptr<const Expr>`.
- Declarations and resources compare by identity.
- Types are `shared_ptr<const TypeInfo>`. Scalar, vector and matrix types
  are interned, so pointer equality is type equality.
- moga calculates the layout of a struct one time and keeps it on the type.
- `Val` overloads the arithmetic, comparison, bitwise and logical operators.
- Equality is `eq` and `ne`, not `==` and `!=`. This prevents the rewritten
  comparisons of C++20.
- A number becomes a literal with the type of the other operand.
- `block["field"]` reads a uniform field or a table record field.
- `Args` gives the function parameters by name or by position.
- Statement and declaration functions take a defaulted
  `std::source_location`. When source locations are on, moga uses it to map
  generated lines to C++ lines (ADR 0004).
- The build state is thread local. It holds the stack of open blocks, the
  current stage and the source location switch.

## Consequences

Positive:

- Shaders look like GLSL, for example `camera["viewProj"] * vec4(p, 1.0)`.
- The compiler makes sure that each visitor handles each node kind.
- Different threads can build programs at the same time.

Negative:

- `&&` and `||` do not short circuit. They always build the two operands.
- moga checks field names when the shader is built, not when C++ compiles.

## Related

- ADR 0002: single source shader DSL.
- ADR 0004: build entry point and diagnostics.
