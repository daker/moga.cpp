# 6. Builtin functions

The builtin functions have the GLSL meaning. The backends translate them.
A number argument gets the type of the other arguments.

Most functions take a scalar or a vector, and apply to each component.

## Common math

| Function | Result | Argument types |
| --- | --- | --- |
| `abs(a)` | The absolute value. | Numbers. |
| `sign(a)` | -1, 0 or 1. | Floats, i32. |
| `floor(a)`, `ceil(a)`, `round(a)`, `trunc(a)` | Rounded values. `round` rounds a half to even. | Floats. |
| `fract(a)` | `a - floor(a)`. | Floats. |
| `mod(a, b)` | `a - b * floor(a / b)`. | Floats. For integers, use `%`. |
| `min(a, b)`, `max(a, b)` | The smaller or the larger value. | Numbers. |
| `clamp(x, lo, hi)` | `x` limited to [`lo`, `hi`]. | Numbers. |
| `saturate(a)` | `a` limited to [0, 1]. | Floats. |
| `mix(a, b, t)` | `a * (1 - t) + b * t`. | Floats. |
| `step(edge, x)` | 0 where `x < edge`, else 1. | Floats. |
| `smoothstep(e0, e1, x)` | The Hermite step from `e0` to `e1`. Reversed edges are valid. | Floats. |

## Exponential and power

| Function | Result |
| --- | --- |
| `pow(a, b)` | `a` raised to `b`. |
| `exp(a)`, `exp2(a)` | e or 2 raised to `a`. |
| `log(a)`, `log2(a)` | The natural or the base 2 logarithm. |
| `sqrt(a)`, `inverseSqrt(a)` | The square root, or 1 divided by the square root. |
| `cbrt(a)` | The cube root. |

## Trigonometry

All angles are in radians.

| Function | Result |
| --- | --- |
| `sin`, `cos`, `tan` | Trigonometric functions. |
| `asin`, `acos`, `atan` | Inverse trigonometric functions. |
| `atan2(y, x)` | The angle of the point (x, y), in [-π, π]. |
| `sinh`, `cosh`, `tanh` | Hyperbolic functions. |
| `asinh`, `acosh`, `atanh` | Inverse hyperbolic functions. |
| `radians(a)`, `degrees(a)` | Angle conversions. |

## Geometry

| Function | Result |
| --- | --- |
| `length(a)` | The length. |
| `lengthSq(a)` | `dot(a, a)`. |
| `distance(a, b)` | The distance between two points. |
| `dot(a, b)` | The dot product. Integer vectors are valid. |
| `cross(a, b)` | The cross product of two `vec3f` values. |
| `normalize(a)` | The vector with length 1. |
| `reflect(i, n)` | The reflection of `i` about `n`. |
| `refract(i, n, eta)` | The refraction direction. |
| `faceForward(n, i, nref)` | `n` when `dot(nref, i) < 0`, else `-n`. |
| `transformDirection(d, m)` | `d` transformed by the upper 3x3 of the mat4 `m`, then normalized. |

## Matrices

| Function | Result |
| --- | --- |
| `transpose(m)` | The transpose. |
| `determinant(m)` | The determinant. |
| `inverse(m)` | The inverse. WGSL gets a helper function. |

## Bool vectors

| Function | Result |
| --- | --- |
| `any(v)` | True when one or more components are true. |
| `all(v)` | True when all components are true. |
| `select(c, t, f)` | `t` where `c` is true, else `f`. The two values are evaluated. |

## Derivatives

Use these in the fragment stage only. They need uniform control flow (see
[5. Control flow and functions](05-control-flow-and-functions.md)).

| Function | Result |
| --- | --- |
| `dpdx(a)` | The screen space derivative along x. |
| `dpdy(a)` | The screen space derivative along y. Positive y points down. |
| `fwidth(a)` | `abs(dpdx(a)) + abs(dpdy(a))`. |

## Bits and packing

| Function | Result |
| --- | --- |
| `floatBitsToUint(v)`, `floatBitsToInt(v)` | The bits of an f32 as u32 or i32. |
| `uintBitsToFloat(v)`, `intBitsToFloat(v)` | The bits of a u32 or i32 as f32. |
| `packHalf2x16(v)` | A vec2f as two 16 bit floats in one u32. |
| `packSnorm2x16(v)`, `packUnorm2x16(v)` | A vec2f as two 16 bit normalized values in one u32. |
| `unpackHalf2x16(u)`, `unpackSnorm2x16(u)`, `unpackUnorm2x16(u)` | The opposite operations. |

## Shortcuts

These functions make common expressions shorter.

| Function | Result |
| --- | --- |
| `oneMinus(a)` | `1 - a` |
| `reciprocal(a)` | `1 / a` |
| `negate(a)` | `-a` |
| `pow2(a)`, `pow3(a)`, `pow4(a)` | `a * a`, `a * a * a`, `a * a * a * a` |
| `difference(a, b)` | `abs(a - b)` |
| `remap(x, inLow, inHigh, outLow = 0, outHigh = 1)` | `x` mapped from one range to the other. The result is not clamped. |
| `remapClamp(...)` | The same, clamped to the output range. |

## Constant arguments

When an argument is a constant, moga checks the domain of the function.
For example, these are errors:

- `sqrt` of a negative constant
- `log` of a constant that is 0 or less
- `asin` or `acos` of a constant outside [-1, 1]
- `clamp` with constant bounds where `lo` is more than `hi`
- `smoothstep` with constant edges that can be equal
- `normalize` of a constant zero vector
- a constant result that overflows, for example in `exp`
