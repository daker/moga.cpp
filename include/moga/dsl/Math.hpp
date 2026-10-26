#pragma once

// Builtin math, geometry, derivative and bit functions.

#include "moga/dsl/Value.hpp"

namespace moga {

// Builtin functions. They have the GLSL meaning; the backends translate them.
// Numbers take the type of the other operand.
/** Returns the absolute value of a numeric scalar or vector. */
Val abs(const ValueLike& a);
/** Returns -1, 0 or 1 for each component of a float or i32 scalar or vector. */
Val sign(const ValueLike& a);
/** Rounds each component down. Takes a float scalar or vector. */
Val floor(const ValueLike& a);
/** Rounds each component up. Takes a float scalar or vector. */
Val ceil(const ValueLike& a);
/** Returns a - floor(a). Takes a float scalar or vector. */
Val fract(const ValueLike& a);
/** Rounds to the nearest integer; a half rounds to even on both backends (GLSL roundEven). */
Val round(const ValueLike& a);
/** Rounds each component toward zero. Takes a float scalar or vector. */
Val trunc(const ValueLike& a);
/** Returns the square root. A constant negative argument is an error. */
Val sqrt(const ValueLike& a);
/** Returns 1 / sqrt(a) (GLSL inversesqrt). A constant argument of 0 or less is an error. */
Val inverseSqrt(const ValueLike& a);
/** Returns e raised to `a`. A constant result that overflows is an error. */
Val exp(const ValueLike& a);
/** Returns 2 raised to `a`. A constant result that overflows is an error. */
Val exp2(const ValueLike& a);
/** Returns the natural logarithm. A constant argument of 0 or less is an error. */
Val log(const ValueLike& a);
/** Returns the base 2 logarithm. A constant argument of 0 or less is an error. */
Val log2(const ValueLike& a);
/** Returns the sine of an angle in radians. */
Val sin(const ValueLike& a);
/** Returns the cosine of an angle in radians. */
Val cos(const ValueLike& a);
/** Returns the tangent of an angle in radians. */
Val tan(const ValueLike& a);
/** Returns the arc sine in radians. A constant argument outside [-1, 1] is an error. */
Val asin(const ValueLike& a);
/** Returns the arc cosine in radians. A constant argument outside [-1, 1] is an error. */
Val acos(const ValueLike& a);
/** Returns the arc tangent in radians, in [-PI/2, PI/2]. Use atan2() for a full turn. */
Val atan(const ValueLike& a);
/** Clamps each component to [0, 1]. Takes a float scalar or vector. */
Val saturate(const ValueLike& a);
/** Returns the vector with length 1. Takes a float vector. A constant zero vector is an error. */
Val normalize(const ValueLike& a);
/** Returns the screen space derivative along x (GLSL dFdx). Takes f32 values. Fragment stage only. */
Val dpdx(const ValueLike& a);
/** Returns the screen space derivative along y (GLSL dFdy). Positive y points down. Fragment stage only. */
Val dpdy(const ValueLike& a);
/** Returns abs(dpdx(a)) + abs(dpdy(a)). Takes f32 values. Fragment stage only. */
Val fwidth(const ValueLike& a);

/** Returns the smaller value per component. A scalar operand widens to the vector type. */
Val min(const ValueLike& a, const ValueLike& b);
/** Returns the larger value per component. A scalar operand widens to the vector type. */
Val max(const ValueLike& a, const ValueLike& b);
/** Returns `a` raised to `b`. Takes floats. A constant negative `a`, or 0 raised to 0 or less, is an error. */
Val pow(const ValueLike& a, const ValueLike& b);
/** Returns the angle of the point (`b`, `a`) in radians, in [-PI, PI] (GLSL atan(y, x)). `a` is y. */
Val atan2(const ValueLike& a, const ValueLike& b);
/** Returns a - b * floor(a / b). For an integer remainder, use `%`. */
Val mod(const ValueLike& a, const ValueLike& b);
/** Returns 0 where `x` < `edge`, else 1. Takes floats; a scalar edge widens to the vector type. */
Val step(const ValueLike& edge, const ValueLike& x);
/** Clamps `x` to [`lo`, `hi`]. Constant bounds with `lo` above `hi` are an error. */
Val clamp(const ValueLike& x, const ValueLike& lo, const ValueLike& hi);
/** Returns a * (1 - t) + b * t. Takes floats; a scalar `t` widens to the vector type. */
Val mix(const ValueLike& a, const ValueLike& b, const ValueLike& t);
/**
 * Returns the smooth Hermite step of `x` between the edges `e0` and `e1`.
 * Reversed edges are valid. Edges that can be the same constant are an error.
 */
Val smoothstep(const ValueLike& e0, const ValueLike& e1, const ValueLike& x);
/** Returns the length of a float scalar or vector. */
Val length(const ValueLike& a);
/** Returns the distance between two points of the same float type. */
Val distance(const ValueLike& a, const ValueLike& b);
/** Returns the dot product of two numeric vectors of the same type. */
Val dot(const ValueLike& a, const ValueLike& b);
/** Returns the cross product of two vec3f values. */
Val cross(const Val& a, const Val& b);
/** Returns the reflection of the incident vector `i` about the normal `n`. Takes float vectors. */
Val reflect(const Val& i, const Val& n);
/** Returns `n` when dot(`nref`, `i`) is negative, else -`n`. Takes float vectors of one type. */
Val faceForward(const Val& n, const Val& i, const Val& nref);
/** Returns the transpose of a matrix. */
Val transpose(const Val& m);
/** Returns the determinant of a square matrix. */
Val determinant(const Val& m);
/** Returns true when any component of a bool or bool vector is true. */
Val any(const Val& v);
/** Returns true when all components of a bool or bool vector are true. */
Val all(const Val& v);
/** Returns the bits of an f32 scalar or vector as u32 (WGSL bitcast). */
Val floatBitsToUint(const Val& v);
/** Returns the bits of a u32 scalar or vector as f32 (WGSL bitcast). */
Val uintBitsToFloat(const Val& v);
/** Returns the bits of an f32 scalar or vector as i32 (WGSL bitcast). */
Val floatBitsToInt(const Val& v);
/** Returns the bits of an i32 scalar or vector as f32 (WGSL bitcast). */
Val intBitsToFloat(const Val& v);

// Hyperbolic functions.
/** Returns the hyperbolic sine. A constant result that overflows is an error. */
Val sinh(const ValueLike& a);
/** Returns the hyperbolic cosine. A constant result that overflows is an error. */
Val cosh(const ValueLike& a);
/** Returns the hyperbolic tangent. */
Val tanh(const ValueLike& a);
/** Returns the inverse hyperbolic sine. */
Val asinh(const ValueLike& a);
/** Returns the inverse hyperbolic cosine. A constant argument below 1 is an error. */
Val acosh(const ValueLike& a);
/** Returns the inverse hyperbolic tangent. A constant argument outside (-1, 1) is an error. */
Val atanh(const ValueLike& a);
/** Converts degrees to radians. */
Val radians(const ValueLike& a);
/** Converts radians to degrees. */
Val degrees(const ValueLike& a);
/** Returns the refraction direction for an incident vector, a normal and a ratio of indices. */
Val refract(const Val& i, const Val& n, const ValueLike& eta);
/** Returns the inverse of a square matrix. WGSL gets a helper function. */
Val inverse(const Val& m);
/** Returns the cube root. Negative values give negative roots. */
Val cbrt(const ValueLike& a);

/** Returns 1 - a. */
Val oneMinus(const ValueLike& a);
/** Returns 1 / a. */
Val reciprocal(const ValueLike& a);
/** Returns -a. */
Val negate(const Val& a);
/** Returns a * a. */
Val pow2(const ValueLike& a);
/** Returns a * a * a. */
Val pow3(const ValueLike& a);
/** Returns a * a * a * a. */
Val pow4(const ValueLike& a);
/** Returns the squared length, dot(a, a). */
Val lengthSq(const Val& a);
/** Returns abs(a - b). */
Val difference(const ValueLike& a, const ValueLike& b);
/** Maps x from [inLow, inHigh] to [outLow, outHigh]. The result is not clamped. */
Val remap(const ValueLike& x, const ValueLike& inLow, const ValueLike& inHigh,
          const ValueLike& outLow = 0, const ValueLike& outHigh = 1);
/** Maps x from [inLow, inHigh] to [outLow, outHigh] and clamps to that range. */
Val remapClamp(const ValueLike& x, const ValueLike& inLow, const ValueLike& inHigh,
               const ValueLike& outLow = 0, const ValueLike& outHigh = 1);
/** Transforms a direction by the upper 3x3 of a mat4 and normalizes the result. */
Val transformDirection(const Val& direction, const Val& matrix);

// Packs two floats into one u32, and unpacks them.
/** Packs a vec2f into a u32 as two 16 bit floats (WGSL pack2x16float). */
Val packHalf2x16(const Val& v);
/** Packs a vec2f in [-1, 1] into a u32 as two 16 bit signed normalized values. */
Val packSnorm2x16(const Val& v);
/** Packs a vec2f in [0, 1] into a u32 as two 16 bit unsigned normalized values. */
Val packUnorm2x16(const Val& v);
/** Unpacks a u32 of two 16 bit floats into a vec2f (WGSL unpack2x16float). */
Val unpackHalf2x16(const Val& v);
/** Unpacks a u32 of two 16 bit signed normalized values into a vec2f. */
Val unpackSnorm2x16(const Val& v);
/** Unpacks a u32 of two 16 bit unsigned normalized values into a vec2f. */
Val unpackUnorm2x16(const Val& v);

/**
 * Returns t when cond is true, else f. A bool vector selects each component.
 * Both t and f are evaluated, on every backend (as WGSL select() does); use If
 * to run only one of two computations.
 */
Val select(const ValueLike& cond, const ValueLike& t, const ValueLike& f);

} // namespace moga
