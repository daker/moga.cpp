#pragma once

// Shader value types. Scalars, vectors and matrices are interned singletons,
// so equality is identity for them; arrays and structs compare structurally.

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moga {

/** A C++ source location, of a DSL statement or of the code that caused an error. */
struct DebugInfo {
  /** The source file path. */
  std::string file;
  /** The line number. */
  std::uint32_t line = 0;
  /** The column number. */
  std::uint32_t column = 0;
};

/** The kind of a shader error. Tests and tools match on it; the message gives the details. */
enum class ErrorCode {
  /** A value has the wrong type for where it is used. */
  TypeMismatch,
  /** An operator, swizzle, member access or index does not apply to its operands. */
  InvalidOperands,
  /** A call or constructor has the wrong number of arguments or components. */
  ArgumentCount,
  /** A type that is not valid in this place (void, atomic, bool in memory, bad size). */
  InvalidType,
  /** A field, parameter or swizzle name that does not exist. */
  UnknownName,
  /** Two declarations, fields or case values with the same name or value. */
  DuplicateName,
  /** A name that collides with a name moga reserves. */
  ReservedName,
  /** Something used in a stage, or a function, where it is not valid. */
  WrongStage,
  /** A write to something that cannot be written. */
  NotAssignable,
  /** A constant, index or literal outside its valid range. */
  OutOfRange,
  /** A value that must be a compile time constant is not. */
  ConstantRequired,
  /** A constant computation that is undefined (division by zero, overflow, non finite). */
  InvalidConstant,
  /** Control flow that is malformed (Else twice, Break outside a loop, missing Return). */
  ControlFlow,
  /** The stages do not fit together (varyings, outputs, Position). */
  StageInterface,
  /** A memory layout that is invalid or differs between backends. */
  Layout,
  /** A texture used in a way its declaration does not allow. */
  TextureUsage,
  /** Declaration options that are invalid together. */
  InvalidDeclaration,
  /** A function calls itself. */
  Recursion,
  /** A barrier or derivative that not all invocations reach together. */
  Uniformity,
  /** A local used outside the block that declares it. */
  Scope,
  /** The target lacks a feature the program uses. */
  MissingFeature,
  /** A statement outside any stage or function body. */
  OutsideStage,
  /** A UniformWriter call that does not match its layout. */
  UniformWriter,
  /** A GPU driver rejected the generated code (from a compile log). */
  Driver,
  /** A fault inside moga; please report it. */
  Internal,
};

/** Returns the name of an error code, such as "TypeMismatch". */
std::string_view errorCodeName(ErrorCode code);

/**
 * An error in a shader program. When it is found while C++ code builds the
 * program, it carries the line and column of that code (the DSL operand or
 * call that was being built), and what() starts with "file:line:column: ".
 */
class ShaderError : public std::runtime_error {
public:
  /** Makes an error without a location. */
  ShaderError(ErrorCode code, const std::string& message);
  /**
   * An error about the operand at this location, when it is in user code;
   * otherwise the location of the code that was building the program.
   */
  ShaderError(ErrorCode code, const std::string& message, const std::source_location& at);
  /** An error found after building, at a statement recorded with source locations. */
  ShaderError(ErrorCode code, const std::string& message, std::optional<DebugInfo> where);
  /** Returns the kind of the error. */
  ErrorCode code() const { return code_; }
  /** The C++ location of the code that caused the error, if known. */
  const std::optional<DebugInfo>& where() const { return where_; }
  /** The message without the location. */
  const std::string& message() const { return message_; }

private:
  ErrorCode code_;
  std::string message_;
  std::optional<DebugInfo> where_;
};

namespace detail {
/**
 * Records where user code passed a value to the DSL. Locations inside moga's
 * own sources and headers are ignored, so an error points at the user's line.
 */
void noteUserLocation(const std::source_location& loc);
/** Forgets the recorded location: a statement is complete, or building is over. */
void clearUserLocation();
} // namespace detail

/** The component kind of a scalar, vector or matrix. F16 needs the shader-f16 device feature and is WebGPU only. */
enum class ScalarKind { F32, I32, U32, Bool, F16 };
/** The kind of a shader type. */
enum class TypeKind { Scalar, Vector, Matrix, Array, Struct, Atomic, Void };

struct TypeInfo;
struct StructLayout;
/** A shared handle to a shader type. Compare types with typesEqual(). */
using TypePtr = std::shared_ptr<const TypeInfo>;

/** One field of a struct type. */
struct StructField {
  /** The field name. */
  std::string name;
  /** The field type. */
  TypePtr type;
};

/** A shader type. Make it with the type functions below, not directly. */
struct TypeInfo {
  /** The kind of the type. */
  TypeKind kind = TypeKind::Void;
  /** The component kind of a scalar, vector, matrix or atomic. */
  ScalarKind scalar = ScalarKind::F32;
  /** Vector size. */
  int size = 0;
  /** Matrix columns; `scalar` is F32 or F16. */
  int cols = 0;
  /** Matrix rows. */
  int rows = 0;
  /** Array element type. */
  TypePtr element;
  /** Array length. */
  int length = 0;
  /** Struct name. */
  std::string name;
  /** Struct fields, in declaration order. */
  std::vector<StructField> fields;

  // Uniform layout of a struct, calculated on first use (see Layout.hpp).
  mutable std::once_flag layoutOnce;
  mutable std::shared_ptr<const StructLayout> layout;
};

/** The fields of a struct, uniform block or table, or the parameters of a function: (name, type) pairs in order. */
using StructFields = std::vector<std::pair<std::string, TypePtr>>;

/** Returns the 32 bit float scalar type. */
const TypePtr& f32();
/** Returns the 32 bit signed integer scalar type. */
const TypePtr& i32();
/** Returns the 32 bit unsigned integer scalar type. */
const TypePtr& u32();
/** Returns the boolean scalar type. */
const TypePtr& boolType();
/** Returns the 16 bit float scalar type. WebGPU only; needs shader-f16. */
const TypePtr& f16();
// Vector types: vecNf (float), vecNi (int), vecNu (uint), vecNb (bool).
/** Returns the vec2f type (GLSL vec2). */
const TypePtr& vec2f();
/** Returns the vec3f type (GLSL vec3). */
const TypePtr& vec3f();
/** Returns the vec4f type (GLSL vec4). */
const TypePtr& vec4f();
/** Returns the vec2i type (GLSL ivec2). */
const TypePtr& vec2i();
/** Returns the vec3i type (GLSL ivec3). */
const TypePtr& vec3i();
/** Returns the vec4i type (GLSL ivec4). */
const TypePtr& vec4i();
/** Returns the vec2u type (GLSL uvec2). */
const TypePtr& vec2u();
/** Returns the vec3u type (GLSL uvec3). */
const TypePtr& vec3u();
/** Returns the vec4u type (GLSL uvec4). */
const TypePtr& vec4u();
/** Returns the vec2<bool> type (GLSL bvec2). */
const TypePtr& vec2b();
/** Returns the vec3<bool> type (GLSL bvec3). */
const TypePtr& vec3b();
/** Returns the vec4<bool> type (GLSL bvec4). */
const TypePtr& vec4b();
// 16 bit float vectors (vecNh). WebGPU only; need shader-f16.
/** Returns the vec2h type. WebGPU only; needs shader-f16. */
const TypePtr& vec2h();
/** Returns the vec3h type. WebGPU only; needs shader-f16. */
const TypePtr& vec3h();
/** Returns the vec4h type. WebGPU only; needs shader-f16. */
const TypePtr& vec4h();
// Square float matrix types.
/** Returns the mat2x2f type (GLSL mat2). */
const TypePtr& mat2f();
/** Returns the mat3x3f type (GLSL mat3). */
const TypePtr& mat3f();
/** Returns the mat4x4f type (GLSL mat4). */
const TypePtr& mat4f();
// Square 16 bit float matrix types. WebGPU only; need shader-f16.
/** Returns the mat2x2h type. WebGPU only; needs shader-f16. */
const TypePtr& mat2h();
/** Returns the mat3x3h type. WebGPU only; needs shader-f16. */
const TypePtr& mat3h();
/** Returns the mat4x4h type. WebGPU only; needs shader-f16. */
const TypePtr& mat4h();
/** Returns the type of a function that returns no value. */
const TypePtr& voidType();
/** Returns the atomic u32 type. Use it in read/write storage buffers and workgroup arrays. */
const TypePtr& atomicU32();
/** Returns the atomic i32 type. Use it in read/write storage buffers and workgroup arrays. */
const TypePtr& atomicI32();

/** Returns the scalar type of this kind. */
const TypePtr& scalarType(ScalarKind kind);
/** Returns the vector type of this kind and size. Size 1 returns the scalar type. */
const TypePtr& vectorType(ScalarKind kind, int size);
/** Returns the n x n float matrix type of this float kind. Use n from 2 to 4. */
const TypePtr& squareMatrix(int n, ScalarKind kind = ScalarKind::F32);
/** Makes a fixed length array type. */
TypePtr arrayOf(TypePtr element, int length, std::source_location loc = std::source_location::current());
/** Makes a struct type. The fields keep the given order. */
TypePtr makeStruct(std::string name, const StructFields& fields, std::source_location loc = std::source_location::current());

/** Returns true when the two types are equal. Structs compare by name and fields. */
bool typesEqual(const TypePtr& a, const TypePtr& b);
/** Returns the component kind of a scalar, vector or matrix type. */
ScalarKind scalarOf(const TypePtr& t);
/** Returns the number of components of a scalar, vector or matrix type. */
int componentCount(const TypePtr& t);
/** Returns true for int, uint and float scalars and vectors. */
bool isNumeric(const TypePtr& t);
/** Returns true for float scalars, vectors and matrices (f32 or f16). */
bool isFloat(const TypePtr& t);
/** Returns true for F32 and F16. */
bool isFloatKind(ScalarKind k);
/** Returns true when the type is or holds an f16 value. */
bool containsF16(const TypePtr& t);
/** Returns true for int and uint scalars and vectors. */
bool isInteger(const TypePtr& t);
/** Returns true for bool scalars and vectors. */
bool isBool(const TypePtr& t);
/** Returns true for an atomic type or a struct or array that holds one. */
bool containsAtomic(const TypePtr& t);
/** Returns the WGSL name of the type. Use it in messages. */
std::string typeName(const TypePtr& t);

} // namespace moga
