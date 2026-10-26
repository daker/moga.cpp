#pragma once

// Vector, matrix, array and struct constructors and conversions.

#include "moga/dsl/Value.hpp"

#include <vector>

namespace moga {

/**
 * Makes a value of a type. For a scalar, vector or matrix, give components;
 * one scalar fills all components. For a struct, give one value for each
 * field, in order. For an array, give one value for each element.
 */
Val construct(const TypePtr& type, const std::vector<ValueLike>& args);

/** Makes an array value. All elements must have the same type. */
Val array(const std::vector<ValueLike>& elements);

// Constructors for common types, as construct(). The first argument is a
// ValueLike so that converting it records the line of the call.

/** Makes a vec2f (GLSL vec2) from components or vectors; one scalar fills both components. */
template <class... A> Val vec2(const ValueLike& first, const A&... rest) { return construct(vec2f(), {first, ValueLike(rest)...}); }
/** Makes a vec3f (GLSL vec3) from components or vectors; one scalar fills all components. */
template <class... A> Val vec3(const ValueLike& first, const A&... rest) { return construct(vec3f(), {first, ValueLike(rest)...}); }
/** Makes a vec4f (GLSL vec4) from components or vectors; one scalar fills all components. */
template <class... A> Val vec4(const ValueLike& first, const A&... rest) { return construct(vec4f(), {first, ValueLike(rest)...}); }
/** Makes a vec2i (GLSL ivec2). */
template <class... A> Val ivec2(const ValueLike& first, const A&... rest) { return construct(vec2i(), {first, ValueLike(rest)...}); }
/** Makes a vec3i (GLSL ivec3). */
template <class... A> Val ivec3(const ValueLike& first, const A&... rest) { return construct(vec3i(), {first, ValueLike(rest)...}); }
/** Makes a vec4i (GLSL ivec4). */
template <class... A> Val ivec4(const ValueLike& first, const A&... rest) { return construct(vec4i(), {first, ValueLike(rest)...}); }
/** Makes a vec2u (GLSL uvec2). */
template <class... A> Val uvec2(const ValueLike& first, const A&... rest) { return construct(vec2u(), {first, ValueLike(rest)...}); }
/** Makes a vec3u (GLSL uvec3). */
template <class... A> Val uvec3(const ValueLike& first, const A&... rest) { return construct(vec3u(), {first, ValueLike(rest)...}); }
/** Makes a vec4u (GLSL uvec4). */
template <class... A> Val uvec4(const ValueLike& first, const A&... rest) { return construct(vec4u(), {first, ValueLike(rest)...}); }
/** Makes a bool vector of 2 components (GLSL bvec2). */
template <class... A> Val bvec2(const ValueLike& first, const A&... rest) { return construct(vec2b(), {first, ValueLike(rest)...}); }
/** Makes a bool vector of 3 components (GLSL bvec3). */
template <class... A> Val bvec3(const ValueLike& first, const A&... rest) { return construct(vec3b(), {first, ValueLike(rest)...}); }
/** Makes a bool vector of 4 components (GLSL bvec4). */
template <class... A> Val bvec4(const ValueLike& first, const A&... rest) { return construct(vec4b(), {first, ValueLike(rest)...}); }
/** Makes a mat2x2f from one diagonal value, 4 components or 2 columns. */
template <class... A> Val mat2(const ValueLike& first, const A&... rest) { return construct(mat2f(), {first, ValueLike(rest)...}); }
/** Makes a mat3x3f from one diagonal value, 9 components, 3 columns, or the upper left of a mat4x4f. */
template <class... A> Val mat3(const ValueLike& first, const A&... rest) { return construct(mat3f(), {first, ValueLike(rest)...}); }
/** Makes a mat4x4f from one diagonal value, 16 components (column major) or 4 columns. */
template <class... A> Val mat4(const ValueLike& first, const A&... rest) { return construct(mat4f(), {first, ValueLike(rest)...}); }
// f16 vectors and matrices; WebGPU only, they need the shader-f16 device feature
/** Makes a vec2h. WebGPU only; needs shader-f16. */
template <class... A> Val hvec2(const ValueLike& first, const A&... rest) { return construct(vec2h(), {first, ValueLike(rest)...}); }
/** Makes a vec3h. WebGPU only; needs shader-f16. */
template <class... A> Val hvec3(const ValueLike& first, const A&... rest) { return construct(vec3h(), {first, ValueLike(rest)...}); }
/** Makes a vec4h. WebGPU only; needs shader-f16. */
template <class... A> Val hvec4(const ValueLike& first, const A&... rest) { return construct(vec4h(), {first, ValueLike(rest)...}); }
/** Makes a mat2x2h. WebGPU only; needs shader-f16. */
template <class... A> Val hmat2(const ValueLike& first, const A&... rest) { return construct(mat2h(), {first, ValueLike(rest)...}); }
/** Makes a mat3x3h. WebGPU only; needs shader-f16. */
template <class... A> Val hmat3(const ValueLike& first, const A&... rest) { return construct(mat3h(), {first, ValueLike(rest)...}); }
/** Makes a mat4x4h. WebGPU only; needs shader-f16. */
template <class... A> Val hmat4(const ValueLike& first, const A&... rest) { return construct(mat4h(), {first, ValueLike(rest)...}); }
/** Makes an RGB color. It is the same as vec3(). */
template <class... A> Val color(const ValueLike& first, const A&... rest) { return construct(vec3f(), {first, ValueLike(rest)...}); }

/** Converts to f32. A number becomes an f32 literal. */
Val toF32(const ValueLike& a);
/** Converts to i32. A number becomes an i32 literal. */
Val toI32(const ValueLike& a);
/** Converts a scalar to f16 (a number becomes an f16 literal). WebGPU only. */
Val toF16(const ValueLike& a);
/** Converts to u32. A number becomes a u32 literal. */
Val toU32(const ValueLike& a);
/** Converts to bool. Zero is false. A literal becomes a bool literal. */
Val toBool(const ValueLike& a);
/** Converts a scalar or vector to another component kind. The size does not change. */
Val cast(const Val& v, ScalarKind scalar);

/**
 * Converts a scalar or vector to a scalar or vector type. A scalar fills all
 * components. A larger vector keeps its first components. A smaller vector
 * gets zero components at the end.
 */
Val convert(const Val& v, const TypePtr& type);

// Conversions to a vector type, as convert().
/** Converts a scalar or vector to vec2f, as convert(). */
inline Val toVec2(const Val& v) { return convert(v, vec2f()); }
/** Converts a scalar or vector to vec3f, as convert(). */
inline Val toVec3(const Val& v) { return convert(v, vec3f()); }
/** Converts a scalar or vector to vec4f, as convert(). */
inline Val toVec4(const Val& v) { return convert(v, vec4f()); }
/** Converts a scalar or vector to vec2i, as convert(). */
inline Val toIVec2(const Val& v) { return convert(v, vec2i()); }
/** Converts a scalar or vector to vec3i, as convert(). */
inline Val toIVec3(const Val& v) { return convert(v, vec3i()); }
/** Converts a scalar or vector to vec4i, as convert(). */
inline Val toIVec4(const Val& v) { return convert(v, vec4i()); }
/** Converts a scalar or vector to vec2u, as convert(). */
inline Val toUVec2(const Val& v) { return convert(v, vec2u()); }
/** Converts a scalar or vector to vec3u, as convert(). */
inline Val toUVec3(const Val& v) { return convert(v, vec3u()); }
/** Converts a scalar or vector to vec4u, as convert(). */
inline Val toUVec4(const Val& v) { return convert(v, vec4u()); }
/** Converts a scalar or vector to a 2 component bool vector, as convert(). */
inline Val toBVec2(const Val& v) { return convert(v, vec2b()); }
/** Converts a scalar or vector to a 3 component bool vector, as convert(). */
inline Val toBVec3(const Val& v) { return convert(v, vec3b()); }
/** Converts a scalar or vector to a 4 component bool vector, as convert(). */
inline Val toBVec4(const Val& v) { return convert(v, vec4b()); }

} // namespace moga
