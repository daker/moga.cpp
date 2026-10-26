#pragma once

// Named shader constants and the common f32 constants.

#include "moga/dsl/Value.hpp"

#include <string>

namespace moga {

/**
 * Declares a named shader constant. Build the value from literals, constructors,
 * operators, builtin functions and other constants only. Declare it one time
 * and use it in many programs.
 */
Val constant(std::string name, const ValueLike& value, Loc loc = Loc::current());

/** Common f32 constants. The generated code uses the names in capitals. */
namespace consts {
/** PI: the ratio of a circle circumference to its diameter. */
const Val& pi();
/** TWO_PI: 2 * PI. */
const Val& twoPi();
/** HALF_PI: PI / 2. */
const Val& halfPi();
/** EPSILON: the difference between 1 and the next f32 value. */
const Val& epsilon();
/** INFINITY: positive f32 infinity. */
const Val& infinity();
} // namespace consts

} // namespace moga
