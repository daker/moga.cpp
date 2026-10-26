#pragma once

// Rounding that does not depend on the host floating point environment, and
// IEEE 754 binary16 (f16) rounding and encoding.

#include <cstdint>

namespace moga::detail {

/**
 * Returns v rounded to the nearest whole number, halfway values to even
 * (WGSL round()). The result is the same in every host rounding mode.
 */
double roundHalfToEven(double v);

/**
 * Returns the f16 value nearest to v, ties to even. Values that round past
 * the largest f16 (65504) become infinity; NaN stays NaN.
 */
double roundToHalf(double v);

/**
 * Returns the largest f16 value not above v (up = false) or the smallest not
 * below it (up = true). Past the largest f16 the result is that value or an
 * infinity, by direction.
 */
double halfToward(double v, bool up);

/** Returns the binary16 bits of the f16 value nearest to v. */
std::uint16_t halfBits(double v);

} // namespace moga::detail
