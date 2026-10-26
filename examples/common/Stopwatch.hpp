#pragma once

// Records named laps and prints how long each step of a variant took.

#include <chrono>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class Stopwatch {
public:
  /** Time since the previous lap (or construction) is recorded under `name`. */
  void lap(std::string name) {
    const auto now = Clock::now();
    laps_.emplace_back(std::move(name), std::chrono::duration<double, std::milli>(now - last_).count());
    last_ = now;
  }

  void print(std::string_view variant) const {
    double total = 0;
    std::string parts;
    for (const auto& [name, ms] : laps_) {
      parts += std::format("{}{} {:.3f}", parts.empty() ? "" : ", ", name, ms);
      total += ms;
    }
    std::println("// {} generated in {:.3f} ms ({})", variant, total, parts);
  }

private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point last_ = Clock::now();
  std::vector<std::pair<std::string, double>> laps_;
};
