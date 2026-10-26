#pragma once

// One row per builtin function: names, argument domain, result rule, stages
// and evaluation flags (ADR 0005). The DSL, analysis and both backends read
// their builtin rules from here.

#include "moga/Nodes.hpp"

#include <string_view>

namespace moga::detail {

/** The types that the first argument may have (from the WGSL overloads). */
enum class Domain {
  None,
  FloatScalarOrVector,
  /** f32 only: derivatives, which WGSL has no f16 overload for. */
  F32ScalarOrVector,
  FloatVector,
  Vec3f,
  Vec2f,
  U32,
  Numeric,
  NumericVector,
  Integer,
  Bool,
  BoolScalarOrVector,
  Matrix,
  SquareMatrix,
  F32Bits,
  U32Bits,
  I32Bits,
  /** Float or i32 scalars and vectors: sign(). */
  SignedNumeric
};

/** How the result type follows from the first argument. */
enum class Result { SameAsFirst, F32, U32, Vec2f, Vec4u, Bool, ScalarOfFirst, ShapeU32, ShapeF32, ShapeI32 };

/** The stages that may call a builtin. Any includes function bodies. */
enum class Stages { Any, Fragment, ComputeOrFragment };

struct BuiltinSpec {
  BuiltinFunction fn;
  std::string_view name;
  std::string_view glsl;
  std::string_view wgsl;
  Domain domain;
  Result result;
  Stages stages = Stages::Any;
  /** WGSL can evaluate it at compile time, so it may initialize a constant. */
  bool constEval = true;
  bool subgroup = false;
  bool bitcast = false;
};

/** The row of a builtin. */
const BuiltinSpec& builtinSpec(BuiltinFunction fn);

/** Returns true when the type is in the domain. */
bool inDomain(Domain domain, const TypePtr& t);

/** Returns a short description of a domain, for messages. */
std::string_view domainText(Domain domain);

/** Throws unless the first argument is in the domain; returns the result type. */
TypePtr checkBuiltin(BuiltinFunction fn, const TypePtr& first);

} // namespace moga::detail
