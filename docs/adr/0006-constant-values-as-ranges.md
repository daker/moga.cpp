# 0006. Evaluate constant expressions as ranges of permitted values

- Status: Accepted
- Date: 2026-10-24
- Revised: 2026-10-26
- Topic: Build pipeline

## Context

WGSL rejects a shader when a constant expression breaks a rule. Examples:

- an integer divisor of zero
- an index out of range
- reversed `clamp` bounds
- equal `smoothstep` edges
- a texel offset outside -8..7
- a shift of 32 or more
- a float that is not finite

The DSL checks these rules when the program is built. Thus the error shows
the user's C++ line. The shader does not fail later on the device.

To do these checks, the DSL calculates constant values. But WGSL does not
give one fixed value for each constant expression:

- An inexact float result can be either neighbor of the exact value. This
  applies to a literal such as 0.1, to `f32(16777217u)`, to a sum and to a
  product.
- A subnormal result can become zero.

If moga calculates one value (round to nearest, as Chrome does), it accepts
shaders that a different valid implementation rejects.

If moga treats each inexact value as unknown, it loses checks that are true
for the two neighbors. For example, `length(vec2(1e20, 0))` overflows for
the two neighbors. `ivec2(vec2(7.9, 0))` truncates to 7 for the two
neighbors.

## Decision

Each known component is a range `[low, high]` of the values that WGSL can
give it. The code is `KnownRange` and `knownRanges()` in `src/Known.hpp`.

- An inexact float result widens to the two f32 or f16 neighbors.
- A float result that can be subnormal also includes zero.
- An exact result stays one value. TwoSum and fma find out if a result is
  exact.
- A float division widens two more steps on each side, for the 2.5 ULP that
  WGSL permits.
- An operand of an operator or builtin that can be subnormal also includes
  zero, because WGSL can read it as zero.
- Literals, conversions (also matrix conversions) and copies keep an exact
  subnormal value.
- Integer ranges come from float conversions. A range that wraps across 2^32
  covers the full type.
- A check rejects the expression when any value in the range breaks the
  rule. When the value is a range, the message says "can be".
- `knownComponents()` returns values only when each component has one value.

The ranges are independent:

- Two uses of one named constant are the only values that are known to be
  equal (`sameValue()`), because the constant is declared one time.
- moga prints each other expression at each use. Each copy can round in a
  different way.

## Consequences

Positive:

- The DSL never accepts a constant expression that a valid WGSL
  implementation can reject for the rules above.
- Checks that are true for each permitted value continue to work on inexact
  values.
- A float literal that an f32 holds exactly is printed as its exact decimal.
  Thus its conversion in the shader has one result, as the evaluator
  assumes. Only computed results (operations and builtins) can become zero.

Negative:

- moga rejects some valid shaders. The ranges do not track values that
  depend on one uncertain constant. Examples, when `c` can be 0 or 2:
  - `normalize(vec2(c, 2.0 - c))` is rejected, because each component can
    be zero.
  - `(2u - c) << (30u + c)` is rejected.
  - `For(c, c, ...)` is rejected.
- A wrapped range that covers the full type can include zero, also when
  neither piece includes it.
- These cases need a constant with more than one permitted value. Only an
  inexact conversion or calculation gives such a constant.
- Builtin domain and overflow checks (`length`, `pow`, `mix`, ...) run on
  each combination of range ends, for a maximum of 12 uncertain components.
  Above that limit, they run only on the lows, the highs and the largest
  magnitudes.

## Related

- ADR 0005: validation architecture.
