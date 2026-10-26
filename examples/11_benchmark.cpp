// Measures shader generation on programs of growing size. Each size runs
// many times; the table gives the median and the 90th percentile per step,
// so one slow run (cache miss, scheduler) does not change the result.
//
// Each program is a chain of N functions. A function has locals, a loop, a
// branch, a select, an integer dot and a mod of a call (a WGSL helper), so
// every pass has work: build, analysis, lowering and both printers.

#include <moga/Shader.hpp>

// the benchmark times each build step, which only the internal header exposes
#include "Steps.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <print>
#include <string>
#include <vector>

using namespace moga;

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

ProgramAST buildProgram(int functions) {
  const UniformBlock camera = uniformBlock("camera", {{"viewProj", mat4f()}}, ResourceFrequency::Frame);
  const Val position = attribute("position", vec3f());
  const Varying world = varying("world", vec4f());
  const Output color = output(0);

  std::vector<Function> chain;
  for (int i = 0; i < functions; ++i) {
    const Function* previous = chain.empty() ? nullptr : &chain.back();
    chain.push_back(fn(std::format("step{}", i), {{"x", vec4f()}}, vec4f(), [&](const Args& a) {
      const Val x = a["x"];
      const Val scaled = Let(x * 1.5 + vec4(static_cast<double>(i)), "scaled");
      const Val acc = Var(scaled, "acc");
      For(0, 4, [&](const Val& k) {
        acc.set(mix(acc, normalize(acc + vec4(1.0)), 0.25) + vec4(toF32(k)));
      });
      If(acc.x() > 0.0, [&] { acc.set(acc * 0.5); }).Else([&] { acc.set(-acc); });
      const Val picked = Let(select(acc.y() > 0.0, acc, scaled), "picked");
      const Val counts = Let(dot(ivec4(1, 2, 3, 4), ivec4(i, i, 1, 1)), "counts");
      // a call operand makes WGSL evaluate mod through a helper
      const Val prior = previous != nullptr ? mod((*previous)(picked), 2.0) : mod(picked, 2.0);
      Return(prior + vec4(toF32(counts)) * 0.001);
    }));
  }

  return program(
      "bench", std::format("bench|n={}", functions),
      [&] {
        const Val p = Let(vec4(position, 1.0), "p");
        world.set(p);
        Position(camera["viewProj"] * p);
      },
      [&] { color.set(chain.back()(world.value())); });
}

struct Sample {
  std::array<double, 4> steps{};
  double total = 0;
  std::size_t chars = 0;
};

Sample runOnce(int functions) {
  Sample s;
  auto t = Clock::now();
  const ProgramAST ast = buildProgram(functions);
  s.steps[0] = msSince(t);
  t = Clock::now();
  const AnalyzedProgram analyzed = analyzeProgram(ast);
  s.steps[1] = msSince(t);
  t = Clock::now();
  const CompiledGLSL glsl = compileGLSL(analyzed);
  s.steps[2] = msSince(t);
  t = Clock::now();
  const CompiledWGSL wgsl = compileWGSL(analyzed);
  s.steps[3] = msSince(t);
  for (double v : s.steps) s.total += v;
  s.chars = glsl.vertex.size() + glsl.fragment.size() + wgsl.code.size();
  return s;
}

double percentile(std::vector<double> values, double p) {
  std::ranges::sort(values);
  const auto i = static_cast<std::size_t>(p * static_cast<double>(values.size() - 1) + 0.5);
  return values[i];
}

} // namespace

int main() {
  constexpr std::array kSizes{1, 10, 50, 200};
  // stop a size after this many milliseconds or runs, whichever comes first
  constexpr double kBudgetMs = 1500;
  constexpr int kMaxRuns = 200;
  constexpr int kMinRuns = 10;
  constexpr int kWarmupRuns = 3;
  constexpr std::array<const char*, 4> kStepNames{"build", "analyze", "glsl", "wgsl"};

  std::println("{:>5} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "fns", "runs", "chars",
               "build", "analyze", "glsl", "wgsl", "total", "total p90");
  for (int n : kSizes) {
    for (int i = 0; i < kWarmupRuns; ++i) runOnce(n);
    std::vector<Sample> samples;
    const auto start = Clock::now();
    while (static_cast<int>(samples.size()) < kMinRuns ||
           (msSince(start) < kBudgetMs && static_cast<int>(samples.size()) < kMaxRuns))
      samples.push_back(runOnce(n));

    std::array<double, 4> medians{};
    for (std::size_t step = 0; step < kStepNames.size(); ++step) {
      std::vector<double> values;
      for (const Sample& s : samples) values.push_back(s.steps[step]);
      medians[step] = percentile(values, 0.5);
    }
    std::vector<double> totals;
    for (const Sample& s : samples) totals.push_back(s.total);
    std::println("{:>5} {:>5} {:>9} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}", n,
                 samples.size(), samples.front().chars, medians[0], medians[1], medians[2], medians[3],
                 percentile(totals, 0.5), percentile(totals, 0.9));
  }
  std::println("times in ms: median per step, median and 90th percentile of the total");
}
