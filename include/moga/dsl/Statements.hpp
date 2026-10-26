#pragma once

// Local variables and control flow.

#include "moga/dsl/Value.hpp"

#include <functional>
#include <vector>
#include <string>

namespace moga {

/** Declares a variable of this type, set to zero. */
Val Var(const TypePtr& type, std::string name = "v", Loc loc = Loc::current());
/** Declares a variable with an initial value. */
Val Var(const ValueLike& init, std::string name = "v", Loc loc = Loc::current());
/** Declares a constant. The value is calculated one time. */
Val Let(const ValueLike& init, std::string name = "t", Loc loc = Loc::current());

/** Adds else branches to an If statement. */
class IfChain {
public:
  /** Wraps the last If statement. If() makes it for you. */
  explicit IfChain(st::If* tail) : tail_(tail) {}
  /** Adds a branch that runs when the previous conditions are false and `cond` is true. */
  IfChain& ElseIf(const ValueLike& cond, const std::function<void()>& body);
  /** Adds a branch that runs when all conditions are false. */
  void Else(const std::function<void()>& body);

private:
  st::If* tail_;
};

/** Runs `body` when the bool condition is true. Add ElseIf() and Else() branches to the result. */
IfChain If(const ValueLike& cond, const std::function<void()>& body, Loc loc = Loc::current());

/**
 * Runs `body` for i from start while i < end, then adds step to i. The
 * counter type is the type of start (i32 by default), and must be i32 or u32.
 * A constant loop whose counter wraps around is an error.
 */
void For(const ValueLike& start, const ValueLike& end, const std::function<void(const Val&)>& body,
         const ValueLike& step = 1, Loc loc = Loc::current());
/** The range and condition of a For loop. */
struct ForRange {
  /** The first counter value. Its type gives the counter type: i32 or u32. */
  ValueLike start = 0;
  /** The bound that the condition compares with. */
  ValueLike end = 0;
  /** Use a negative step with an i32 counter to count down. */
  ValueLike step = 1;
  /** Compares the counter with end before each iteration: Lt, Le, Gt or Ge. */
  BinaryOp condition = BinaryOp::Lt;
};

/** Runs `body` while `counter <condition> end`. Then adds step to the counter. */
void For(const ForRange& range, const std::function<void(const Val&)>& body,
         Loc loc = Loc::current());
/** Runs `body` for each (i, j) with 0 <= i < rows and 0 <= j < cols. */
void ForGrid(const ValueLike& rows, const ValueLike& cols,
             const std::function<void(const Val&, const Val&)>& body, Loc loc = Loc::current());
/** Runs `body` while the bool condition is true. */
void While(const ValueLike& cond, const std::function<void()>& body, Loc loc = Loc::current());
/** Runs `body` again and again. Use Break() or Return() to stop; a loop with neither is an error. */
void Loop(const std::function<void()>& body, Loc loc = Loc::current());

/** Adds cases to a Switch statement. */
class SwitchChain {
public:
  /** Wraps the Switch statement. Switch() makes it for you. */
  explicit SwitchChain(st::Switch* s) : switch_(s) {}
  /** Adds a case for one or more number literals. The case does not fall through to the next one. A value used twice is an error. */
  SwitchChain& Case(const std::vector<ValueLike>& values, const std::function<void()>& body);
  /** Adds a case for one value. */
  SwitchChain& Case(const ValueLike& value, const std::function<void()>& body);
  /** Adds the branch that runs when no case matches. */
  void Default(const std::function<void()>& body);

private:
  st::Switch* switch_;
};

/** Selects a case by an i32 or u32 value. Add the cases to the result. */
SwitchChain Switch(const ValueLike& selector, Loc loc = Loc::current());
/** Stops the loop, or leaves the switch case. */
void Break(Loc loc = Loc::current());
/** Starts the next iteration of the loop. */
void Continue(Loc loc = Loc::current());
/** Returns from the function or stage. */
void Return(Loc loc = Loc::current());
/** Returns a value from the function. */
void Return(const ValueLike& value, Loc loc = Loc::current());
/** Discards the fragment. Fragment stage only (or a function that the fragment stage calls). */
void Discard(Loc loc = Loc::current());

} // namespace moga
