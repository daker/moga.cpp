#pragma once

// Shader values and operators. A number operand becomes a literal with the
// type of the other operand.

#include "moga/Nodes.hpp"

#include <concepts>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace moga {

/** The C++ location of a DSL call. Leave the default argument; it records the caller's line. */
using Loc = std::source_location;

class ValueLike;

/** Immutable handle to a shader expression. */
class Val {
public:
  /** Makes a handle to an expression. */
  explicit Val(ExprPtr node) : node_(std::move(node)) {}

  /** Returns the expression. */
  const ExprPtr& node() const { return node_; }
  /** Returns the type of the expression. */
  const TypePtr& type() const { return node_->type; }

  /** Selects components, for example `v.swizzle("xzy")`. You can also use `rgba` names. */
  Val swizzle(std::string_view comps) const;
  // Swizzle shortcuts. rgb() is xyz() and a() is w().
  /** Returns component x. */
  Val x() const { return swizzle("x"); }
  /** Returns component y. */
  Val y() const { return swizzle("y"); }
  /** Returns component z. */
  Val z() const { return swizzle("z"); }
  /** Returns component w. */
  Val w() const { return swizzle("w"); }
  /** Returns components x and y. */
  Val xy() const { return swizzle("xy"); }
  /** Returns components x and z. */
  Val xz() const { return swizzle("xz"); }
  /** Returns components y and z. */
  Val yz() const { return swizzle("yz"); }
  /** Returns components z and w. */
  Val zw() const { return swizzle("zw"); }
  /** Returns components x, y and z. */
  Val xyz() const { return swizzle("xyz"); }
  /** Returns the color components r, g and b (the same as xyz()). */
  Val rgb() const { return swizzle("xyz"); }
  /** Returns the alpha component (the same as w()). */
  Val a() const { return swizzle("w"); }

  /** Reads a struct member. An unknown field is an error. */
  Val member(std::string_view field) const;
  /** Reads an array element, a matrix column or a vector component. The index is an i32 or u32 scalar. */
  Val operator[](const ValueLike& index) const;

  /** Compares for equality. Gives bool, or a bool vector for vectors. */
  Val eq(const ValueLike& o) const;
  /** Compares for inequality. Gives bool, or a bool vector for vectors. */
  Val ne(const ValueLike& o) const;

  /**
   * Assigns a value. Use it on a Var, or on a swizzle, element or member of a
   * Var, or on an element of a ReadWrite storage buffer. The value must have
   * the type of the target. Use atomicStore() for atomics.
   */
  void set(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Adds a value to this variable. */
  void addAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Subtracts a value from this variable. */
  void subAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Multiplies this variable by a value. */
  void mulAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Divides this variable by a value. */
  void divAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Sets this integer variable to the remainder of the division by a value. */
  void remAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Applies a bitwise AND with a value. */
  void bitAndAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Applies a bitwise OR with a value. */
  void bitOrAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Applies a bitwise XOR with a value. */
  void bitXorAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Shifts the bits of this variable to the left. */
  void shlAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Shifts the bits of this variable to the right. */
  void shrAssign(const ValueLike& value, Loc loc = Loc::current()) const;
  /** Adds 1 to this variable. */
  void increment(Loc loc = Loc::current()) const;
  /** Subtracts 1 from this variable. */
  void decrement(Loc loc = Loc::current()) const;

private:
  /** Emits `target op= value`, which evaluates the target once. */
  void compound(BinaryOp op, const ValueLike& value, const Loc& loc) const;

  ExprPtr node_;
};

/** A C++ number type that becomes a shader literal: any arithmetic type except bool. */
template <class T>
concept Number = std::is_arithmetic_v<T> && !std::same_as<T, bool>;

/** Anything accepted where a shader value is expected; numbers become literals. */
class ValueLike {
public:
  // each records the caller's line and column, so an error names the user's code
  /** Wraps a shader value. */
  ValueLike(Val v, Loc loc = Loc::current()) : v_(std::move(v)), loc_(loc) { detail::noteUserLocation(loc); }
  /** Wraps a bool literal. */
  ValueLike(bool b, Loc loc = Loc::current()) : v_(b), loc_(loc) { detail::noteUserLocation(loc); }
  /** Wraps a number literal. It gets its type from the other operand, or f32. */
  template <Number T> ValueLike(T n, Loc loc = Loc::current()) : v_(static_cast<double>(n)), loc_(loc) {
    detail::noteUserLocation(loc);
  }

  /** Where the value was given: the start of the operand in the caller's code. */
  const Loc& where() const { return loc_; }

  /** Returns the shader value, or null for a literal. */
  const Val* val() const { return std::get_if<Val>(&v_); }
  /** Returns the number literal, or null. */
  const double* number() const { return std::get_if<double>(&v_); }
  /** Returns the bool literal, or null. */
  const bool* boolean() const { return std::get_if<bool>(&v_); }

private:
  std::variant<Val, double, bool> v_;
  Loc loc_;
};

/** Converts a literal to a Val. The literal gets the scalar kind of `like` when you give it. */
Val toVal(const ValueLike& v, const TypePtr& like = nullptr);
/** Makes a binary operation. Throws when the operand types are not valid. */
Val binary(BinaryOp op, const ValueLike& a, const ValueLike& b);
/** Makes a unary operation. Throws when the operand type is not valid. */
Val unary(UnaryOp op, const Val& a, const Loc& at = Loc::current());

/**
 * The operands of an operator. Converting to them records where each one
 * starts in the caller's code, so an error can name the operand at fault.
 */
struct Operand {
  /** The shader value. */
  const Val& v;
  /** Where the operand starts in the caller's code. */
  Loc loc;
  /** Wraps a shader value and records its location. */
  Operand(const Val& value, Loc where = Loc::current()) : v(value), loc(where) { detail::noteUserLocation(where); }
};
/** A number operand of an operator. It gets the type of the other operand. */
struct Literal {
  /** The number. */
  double value;
  /** Where the number starts in the caller's code. */
  Loc loc;
  /** Wraps a number and records its location. */
  template <Number T>
  Literal(T n, Loc where = Loc::current()) : value(static_cast<double>(n)), loc(where) {
    detail::noteUserLocation(where);
  }
};

#define MOGA_BINARY_OPERATOR(OP, NAME)                                                         \
  inline Val operator OP(Operand a, Operand b) {                                               \
    return binary(BinaryOp::NAME, ValueLike(a.v, a.loc), ValueLike(b.v, b.loc));              \
  }                                                                                            \
  inline Val operator OP(Operand a, Literal b) {                                               \
    return binary(BinaryOp::NAME, ValueLike(a.v, a.loc), ValueLike(b.value, b.loc));          \
  }                                                                                            \
  inline Val operator OP(Literal a, Operand b) {                                               \
    return binary(BinaryOp::NAME, ValueLike(a.value, a.loc), ValueLike(b.v, b.loc));          \
  }

/** Adds. A scalar operand applies to each component of a vector. */
MOGA_BINARY_OPERATOR(+, Add)
/** Subtracts. */
MOGA_BINARY_OPERATOR(-, Sub)
/** Multiplies. With a matrix, it is the linear algebra product. */
MOGA_BINARY_OPERATOR(*, Mul)
/** Divides. An integer division by a constant zero is an error. */
MOGA_BINARY_OPERATOR(/, Div)
/** Returns the integer remainder. Use mod() for floats. */
MOGA_BINARY_OPERATOR(%, Rem)
/** Compares with less than. Gives bool, or a bool vector for vectors. */
MOGA_BINARY_OPERATOR(<, Lt)
/** Compares with less than or equal. Gives bool, or a bool vector for vectors. */
MOGA_BINARY_OPERATOR(<=, Le)
/** Compares with greater than. Gives bool, or a bool vector for vectors. */
MOGA_BINARY_OPERATOR(>, Gt)
/** Compares with greater than or equal. Gives bool, or a bool vector for vectors. */
MOGA_BINARY_OPERATOR(>=, Ge)
/** Applies a bitwise AND to integers. */
MOGA_BINARY_OPERATOR(&, BitAnd)
/** Applies a bitwise OR to integers. */
MOGA_BINARY_OPERATOR(|, BitOr)
/** Applies a bitwise XOR to integers. */
MOGA_BINARY_OPERATOR(^, BitXor)
/** Shifts integer bits to the left. A constant shift must be in 0..31. */
MOGA_BINARY_OPERATOR(<<, Shl)
/** Shifts integer bits to the right. A constant shift must be in 0..31. */
MOGA_BINARY_OPERATOR(>>, Shr)
#undef MOGA_BINARY_OPERATOR

// Both operands are always evaluated: these build an expression, not a branch.
/** Returns the logical AND of bools. Both operands are evaluated. */
inline Val operator&&(const Val& a, const Val& b) { return binary(BinaryOp::And, a, b); }
/** Returns the logical OR of bools. Both operands are evaluated. */
inline Val operator||(const Val& a, const Val& b) { return binary(BinaryOp::Or, a, b); }
/** Negates a float or i32 scalar or vector, or a matrix. A u32 value is an error. */
inline Val operator-(Operand a) { return unary(UnaryOp::Neg, a.v, a.loc); }
/** Returns the logical NOT of a bool. */
inline Val operator!(Operand a) { return unary(UnaryOp::Not, a.v, a.loc); }
/** Returns the bitwise NOT of an integer. */
inline Val operator~(Operand a) { return unary(UnaryOp::BitNot, a.v, a.loc); }

} // namespace moga
