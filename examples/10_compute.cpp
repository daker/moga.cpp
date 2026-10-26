// WebGPU only: compute programs. Three kernels share one particle buffer:
//  - integrate: moves particles and writes their speed into a storage texture
//  - histogram: counts speeds with atomics in workgroup memory, then merges
//  - reduce: sums the speeds of each workgroup with subgroup operations

#include <moga/Shader.hpp>

#include "common/Report.hpp"
#include "common/Stopwatch.hpp"

#include <print>

using namespace moga;

namespace {

/** Prints a build; returns false when it failed. */
bool print(const BuildResult& result, Stopwatch& sw) {
  sw.lap("build");
  if (!report(result)) return false;
  const Reflection& r = result.program->reflection;
  std::println("// ==== {}", r.variantKey);
  printCode(result);
  for (const ReflectedStorageBuffer& b : r.storageBuffers)
    std::println("// storage {} at group {} binding {}: {} bytes per element", b.name, b.group,
                 b.binding, b.element.stride);
  sw.print(r.variantKey);
  return true;
}

} // namespace

int main() {
  const TypePtr particle =
      makeStruct("Particle", {{"position", vec3f()}, {"mass", f32()}, {"velocity", vec3f()}});
  const StorageBuffer particles = storage("particles", particle);
  const StorageBuffer bins = storage("bins", atomicU32());
  const StorageBuffer sums = storage("sums", f32());
  const StorageTexture speedMap =
      storageTexture("speedMap", TextureDimension::D2, TexelFormat::Rgba8Unorm);
  const Val dt = uniform("dt", f32(), ResourceFrequency::Frame);
  constexpr int kBins = 64;

  {
    Stopwatch sw;
    const BuildResult result = buildComputeShader("integrate", "INTEGRATE", {64, 1, 1}, [&] {
      const Val i = Let(globalId().x(), "i");
      If(i >= particles.length(), [] { Return(); });
      const Val p = particles[i];
      const Val v = Let(p.member("velocity") + vec3(0.0, -9.81, 0.0) * dt, "v");
      p.member("velocity").set(v);
      p.member("position").addAssign(v * dt);
      // one texel per particle, 256 per row
      const Val texel = Let(uvec2(i % 256u, i / 256u), "texel");
      speedMap.store(texel, vec4(vec3(saturate(length(v) / 20.0)), 1.0));
    });
    if (!print(result, sw)) return 1;
  }

  {
    Stopwatch sw;
    const Val local = workgroupArray("localBins", atomicU32(), kBins);
    const BuildResult result = buildComputeShader("histogram", "HISTOGRAM", {64, 1, 1}, [&] {
      const Val i = Let(globalId().x(), "i");
      // each invocation clears one bin; 64 invocations, 64 bins
      atomicStore(local[localIndex()], 0u);
      workgroupBarrier();
      If(i < particles.length(), [&] {
        const Val speed = Let(length(particles[i].member("velocity")), "speed");
        const Val bin = Let(min(toU32(speed * 2.0), toU32(kBins - 1)), "bin");
        atomicAdd(local[bin], 1u);
      });
      workgroupBarrier();
      atomicAdd(bins[localIndex()], atomicLoad(local[localIndex()]));
    });
    if (!print(result, sw)) return 1;
  }

  {
    Stopwatch sw;
    const BuildResult result = buildComputeShader("reduce", "REDUCE", {64, 1, 1}, [&] {
      const Val i = Let(globalId().x(), "i");
      const Val speed = Var(0.0, "speed");
      If(i < particles.length(), [&] { speed.set(length(particles[i].member("velocity"))); });
      const Val total = Let(subgroupAdd(speed), "total");
      If(subgroupElect(), [&] {
        sums[workgroupId().x() * (64u / subgroupSize()) + localIndex() / subgroupSize()].set(total);
      });
    });
    if (!print(result, sw)) return 1;
  }
}
