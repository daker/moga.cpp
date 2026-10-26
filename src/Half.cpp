#include "Half.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace moga::detail {

namespace {

// binary16: 10 fraction bits, exponents -14..15, exponent bias 15
constexpr int kFractionBits = 10;
constexpr int kMinExponent = -14;
constexpr int kExponentBias = 15;
constexpr double kMaxHalf = 65504.0; // (2 - 2^-10) * 2^15
constexpr std::uint16_t kSignBit = 0x8000;
constexpr std::uint16_t kInfinity = 0x7c00;
constexpr std::uint16_t kQuietNan = 0x7e00;

} // namespace

double roundHalfToEven(double v) {
  if (!std::isfinite(v)) return v;
  // floor() and the difference are exact, so no step depends on the rounding mode
  const double below = std::floor(v);
  const double fraction = v - below;
  if (fraction < 0.5) return below;
  if (fraction > 0.5) return below + 1.0;
  return std::fmod(below, 2.0) == 0.0 ? below : below + 1.0;
}

double roundToHalf(double v) {
  if (!std::isfinite(v) || v == 0.0) return v;
  const double a = std::abs(v);
  // a is in [2^e, 2^(e+1)); below 2^-14 the spacing stays that of the subnormals
  const int e = std::max(std::ilogb(a), kMinExponent);
  const double quantum = std::ldexp(1.0, e - kFractionBits);
  // dividing by a power of 2 is exact, so this rounds the f16 fraction bits
  const double r = roundHalfToEven(a / quantum) * quantum;
  return std::copysign(r > kMaxHalf ? std::numeric_limits<double>::infinity() : r, v);
}

double halfToward(double v, bool up) {
  if (!std::isfinite(v) || v == 0.0) return v;
  const int e = std::max(std::ilogb(std::abs(v)), kMinExponent);
  const double quantum = std::ldexp(1.0, e - kFractionBits);
  // dividing by a power of 2 is exact, and floor or ceil of the quotient picks the f16 neighbor
  const double r = (up ? std::ceil(v / quantum) : std::floor(v / quantum)) * quantum;
  const double inf = std::numeric_limits<double>::infinity();
  // past the largest f16, rounding away from zero overflows and rounding toward zero stops at it
  if (r > kMaxHalf) return up ? inf : kMaxHalf;
  if (r < -kMaxHalf) return up ? -kMaxHalf : -inf;
  return r;
}

std::uint16_t halfBits(double v) {
  if (std::isnan(v)) return kQuietNan;
  const double r = roundToHalf(v);
  const std::uint16_t sign = std::signbit(r) ? kSignBit : 0;
  const double a = std::abs(r);
  if (std::isinf(a)) return sign | kInfinity;
  if (a == 0.0) return sign;
  const int e = std::ilogb(a);
  if (e < kMinExponent) // subnormal: the fraction counts steps of 2^-24
    return sign | static_cast<std::uint16_t>(a / std::ldexp(1.0, kMinExponent - kFractionBits));
  const auto fraction = static_cast<std::uint16_t>((a / std::ldexp(1.0, e) - 1.0) * (1 << kFractionBits));
  return sign | static_cast<std::uint16_t>((e + kExponentBias) << kFractionBits) | fraction;
}

} // namespace moga::detail
