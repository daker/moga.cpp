#pragma once

// User shader functions.

#include "moga/dsl/Value.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace moga {

/** The parameters of a function body. */
class Args {
public:
  /** Makes the parameter list. fn() makes it for you. */
  Args(std::vector<std::string> names, std::vector<Val> values)
      : names_(std::move(names)), values_(std::move(values)) {}
  /** Returns the parameter at this position. */
  const Val& operator[](std::size_t i) const { return values_.at(i); }
  /** Returns the parameter with this name. Throws if it does not exist. */
  const Val& operator[](std::string_view name) const;

private:
  std::vector<std::string> names_;
  std::vector<Val> values_;
};

/** A shader function. A call to a void function is also added as a statement. */
class Function {
public:
  /** Wraps a function declaration. Use fn() to make one. */
  explicit Function(FunctionPtr decl) : decl_(std::move(decl)) {}
  /** Returns the declaration. */
  const FunctionPtr& decl() const { return decl_; }
  /** Calls the function. Give one argument for each parameter, in order. A wrong count is an error. */
  Val call(const std::vector<ValueLike>& args) const;
  /** Calls the function, as call(). */
  template <class... A> Val operator()(const A&... args) const { return call({ValueLike(args)...}); }

private:
  FunctionPtr decl_;
};

/**
 * Defines a shader function. `params` gives the name and type of each
 * parameter. Use voidType() for no return value. The body runs one time,
 * now. Do not use attributes, varyings, outputs or builtins in the body. A
 * function with a return type that can reach its end without Return() is an
 * error.
 */
Function fn(std::string name, const StructFields& params, TypePtr returnType,
            const std::function<void(const Args&)>& body, Loc loc = Loc::current());

} // namespace moga
