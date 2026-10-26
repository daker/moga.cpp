#pragma once

// Build state shared by the DSL modules: the open block stack, the current
// stage and the debug flag of the calling thread.

#include "moga/dsl/Value.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace moga::detail {

// Ranges of the shader scalar types, as doubles (number literals are doubles).
inline constexpr double kI32Min = std::numeric_limits<std::int32_t>::min();
inline constexpr double kI32Max = std::numeric_limits<std::int32_t>::max();
inline constexpr double kU32Max = std::numeric_limits<std::uint32_t>::max();
inline constexpr double kF32Max = std::numeric_limits<float>::max();
/** The largest finite f16, (2 - 2^-10) * 2^15. */
inline constexpr double kF16Max = 65504.0;

/**
 * Records statement source locations on this thread while it lives, then
 * restores the previous state, also when the build throws.
 */
class SourceLocationScope {
public:
  explicit SourceLocationScope(bool enabled);
  ~SourceLocationScope();
  SourceLocationScope(const SourceLocationScope&) = delete;
  SourceLocationScope& operator=(const SourceLocationScope&) = delete;

private:
  bool previous_;
};

/** An error kept by body level recovery, with the stage that was being built. */
struct RecordedError {
  ShaderError error;
  std::optional<StageName> stage;
};

/**
 * Where body level recovery keeps errors on this thread, or null when errors
 * propagate. When set, an error in a body ends that body only, and recording
 * goes on after it.
 */
std::vector<RecordedError>*& recordedErrors();

/** Returns the number of errors kept so far; 0 without recovery. */
std::size_t recordedErrorCount();

/** The open blocks of this thread. The last one gets new statements. */
std::vector<Block*>& blockStack();
/** The stage that this thread is building, or nothing. */
std::optional<StageName>& currentStage();

/** Return types of the function bodies being built on this thread, innermost last. */
std::vector<TypePtr>& functionReturnTypes();

/** Adds a statement to the innermost open block and returns it. */
Stmt& pushStatement(StmtNode node, const Loc& loc);

/** Runs `body` and returns the statements that it adds. */
template <class F> Block collectBlock(F&& body) {
  Block block;
  blockStack().push_back(&block);
  struct Pop {
    ~Pop() { blockStack().pop_back(); }
  } pop;
  if (std::vector<RecordedError>* errors = recordedErrors()) {
    try {
      body();
    } catch (const ShaderError& e) {
      errors->push_back({e, currentStage()});
    }
  } else {
    body();
  }
  return block;
}

/** Whether a ShaderBuild records the errors of this thread instead of throwing them. */
inline bool recovering() { return recordedErrors() != nullptr; }

/**
 * Throws the error, or records it while a ShaderBuild is open; the caller
 * then goes on as if the check had passed, so the code after it must be safe
 * to run. Declarations use it: they are still made, so later code can use them.
 */
void fail(const ShaderError& error);

/** Runs validation code; an error in it goes to fail(), so the caller goes on after it. */
template <class F> void check(F&& validate) {
  try {
    validate();
  } catch (const ShaderError& e) {
    fail(e);
  }
}

/** Drops the statements made outside any body while recovering; a ShaderBuild does it when it ends. */
void discardOrphans();

/** A placeholder of a type for a value that an error could not make. */
inline Val poison(const TypePtr& type) { return Val(makePoison(type)); }

inline bool poisoned(const Val& v) { return v.node()->poisoned; }
inline bool poisoned(const ValueLike& v) { return v.val() != nullptr && poisoned(*v.val()); }
inline bool poisoned(const std::optional<ValueLike>& v) { return v && poisoned(*v); }
template <class T> bool poisoned(const std::vector<T>& values) {
  return std::ranges::any_of(values, [](const T& v) { return poisoned(v); });
}
/** Other arguments (names, options, locations, callbacks) hold no value. */
template <class T> bool poisoned(const T&) { return false; }

inline TypePtr typeIn(const Val& v) { return v.type(); }
inline TypePtr typeIn(const ValueLike& v) { return v.val() != nullptr ? v.val()->type() : nullptr; }
/** A type argument, such as the type a constructor makes. */
inline TypePtr typeIn(const TypePtr& t) { return t; }
template <class T> TypePtr typeIn(const T&) { return nullptr; }

/**
 * Runs a DSL operation. While a ShaderBuild is open, an error becomes a
 * diagnostic, and the operation returns a poisoned value of the type of its
 * first value input (f32 when it has none), or does nothing for a statement.
 * An error about a poisoned input is not reported: the first error caused it.
 */
template <class F, class... In> auto guard(F&& op, const In&... inputs) -> decltype(op()) {
  using R = decltype(op());
  static_assert(std::is_same_v<R, Val> || std::is_void_v<R>, "guard returns a Val or nothing");
  if (!recovering()) return op();
  try {
    return op();
  } catch (const ShaderError& e) {
    if (!(poisoned(inputs) || ...)) recordedErrors()->push_back({e, currentStage()});
    if constexpr (std::is_same_v<R, Val>) {
      TypePtr type;
      ((type = type ? type : typeIn(inputs)), ...);
      return poison(type ? type : f32());
    }
  }
}

/** The statements around the block being built, for Break() and Continue(). */
enum class Enclosing { Loop, Switch, Function };

/** The enclosing loops, switches and function bodies of this thread, innermost last. */
std::vector<Enclosing>& enclosing();

/** Runs `body` as the body of a loop, a switch case or a function and returns its statements. */
template <class F> Block collectBody(Enclosing kind, F&& body) {
  enclosing().push_back(kind);
  struct Pop {
    ~Pop() { enclosing().pop_back(); }
  } pop;
  return collectBlock(std::forward<F>(body));
}

/**
 * Throws unless the type can be a local or a value: not void, and neither an
 * atomic nor holding one (WGSL keeps atomics in storage and workgroup memory).
 */
void requireValueType(const TypePtr& type, std::string_view what);

/** Throws when this thread does not build `stage`. */
void requireStage(StageName stage, std::string_view what);
/**
 * Returns true while a function body is built outside any stage. Its stage
 * operations (barriers, subgroup functions, Discard()) are then checked by
 * analysis, against the stages that call it.
 */
bool buildingDetachedFunction();
/** Throws when the value does not have this type. */
void checkType(const Val& v, const TypePtr& type, std::string_view what);

/** Makes a local variable with a new id. */
LocalPtr newLocal(std::string name, TypePtr type, bool mutableVar);
/** Returns a read of a local variable. */
Val localVal(const LocalPtr& v);
/** Makes a literal of this type. */
Val literal(std::variant<double, bool> value, const TypePtr& type);
/**
 * Converts a literal to i32 and checks that the value is an i32 or u32 scalar.
 * A float Val stays a float under toVal(), so this check is needed for indices,
 * levels and layers.
 */
Val integerScalar(const ValueLike& v, std::string_view what);

/** Returns true for scalar and vector types. */
bool isScalarOrVector(const TypePtr& t);

/**
 * Returns the first operand of a two operand helper. A Val stays the same; a
 * literal gets the type of the second operand when that operand is a Val.
 */
Val coerceFirst(const ValueLike& a, const ValueLike& b);

} // namespace moga::detail
