#include <doctest/doctest.h>

#include <algorithm>
#include <moga/Shader.hpp>

#include "Steps.hpp"

#include "Print.hpp"

using namespace moga;

namespace {

bool has(const std::string& haystack, std::string_view needle) {
  return haystack.find(needle) != std::string::npos;
}

/** Compiles a fragment body on WGSL and returns the GLSL error message. */
template <class F> std::pair<std::string, std::string> renderBoth(F&& body) {
  const Output out = output(0);
  const ProgramAST ast = program("r", "R", [] { Position(vec4(0.0)); }, [&] { body(out); });
  const AnalyzedProgram a = analyzeProgram(ast);
  std::string glslError;
  try {
    compileGLSL(a);
  } catch (const ShaderError& e) {
    glslError = e.what();
  }
  return {compileWGSL(a).code, glslError};
}

} // namespace

// Expected offsets follow the WGSL storage address space rules, worked out by hand.

TEST_CASE("storage layout packs a scalar after a vec3 and does not pad arrays to 16") {
  const TypePtr particle =
      makeStruct("P", {{"position", vec3f()}, {"mass", f32()}, {"velocity", vec3f()}});
  const StorageLayout l = storageLayout(particle);
  REQUIRE(l.fields.size() == 3);
  CHECK(l.fields[1].offset == 12);
  CHECK(l.fields[2].offset == 16);
  CHECK(l.size == 32);
  CHECK(l.stride == 32);
  CHECK(storageLayout(arrayOf(vec2f(), 3)).size == 24);
  CHECK(storageLayout(mat3f()).size == 48);
  CHECK(storageLayout(makeStruct("S", {{"a", f32()}, {"b", vec2f()}})).size == 16);
  CHECK_THROWS_AS(storageLayout(boolType()), ShaderError);
}

TEST_CASE("compute program with storage, workgroup memory, barriers and atomics") {
  const StorageBuffer input = storage("values", f32(), StorageAccess::Read);
  const StorageBuffer counts = storage("counts", atomicU32());
  const Val local = workgroupArray("local", atomicU32(), 8);
  const ProgramAST ast = computeProgram("c", "C", {8, 4, 1}, [&] {
    const Val i = Let(globalId().x(), "i");
    atomicStore(local[localIndex()], 0u);
    workgroupBarrier();
    If(i < input.length(), [&] { atomicAdd(local[toU32(input[i]) % 8u], 1u); });
    workgroupBarrier();
    atomicAdd(counts[localIndex()], atomicLoad(local[localIndex()]));
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  const CompiledWGSL w = compileWGSL(a);
  CHECK(w.computeEntry == "cs_main");
  CHECK(w.vertexEntry.empty());
  CHECK(has(w.code, "@compute @workgroup_size(8, 4, 1) fn cs_main(cin: CSIn) {"));
  CHECK(has(w.code, "@builtin(global_invocation_id) globalId: vec3u,"));
  CHECK(has(w.code, "@builtin(local_invocation_index) localIndex: u32,"));
  CHECK(has(w.code, "var<storage, read> sb_values: array<f32>;"));
  CHECK(has(w.code, "var<storage, read_write> sb_counts: array<atomic<u32>>;"));
  CHECK(has(w.code, "var<workgroup> wg_local: array<atomic<u32>, 8>;"));
  CHECK(has(w.code, "arrayLength(&sb_values)"));
  CHECK(has(w.code, "atomicStore(&wg_local[cin.localIndex], 0u);"));
  CHECK(has(w.code, "workgroupBarrier();"));
  CHECK(has(w.code, "atomicAdd(&sb_counts[cin.localIndex], atomicLoad(&wg_local[cin.localIndex]))"));
  CHECK_FALSE(has(w.code, "vs_main"));
  CHECK(a.reflection.compute);
  CHECK(a.reflection.workgroupSize == std::array<std::uint32_t, 3>{8, 4, 1});
  REQUIRE(a.reflection.storageBuffers.size() == 2);
  CHECK(a.reflection.storageBuffers[0].element.stride == 4);
}

TEST_CASE("GLSL names the first WebGPU only feature") {
  const ProgramAST ast = computeProgram("k", "K", {1, 1, 1}, [] {});
  CHECK_THROWS_WITH_AS(compileGLSL(analyzeProgram(ast)),
                       doctest::Contains("k: a compute program needs WebGPU; WebGL2 cannot run it"), ShaderError);

  const StorageBuffer data = storage("data", vec4f(), StorageAccess::Read);
  const auto [wgsl, error] = renderBoth([&](const Output& out) { out.set(data[0u]); });
  CHECK(has(wgsl, "var<storage, read> sb_data: array<vec4f>;"));
  CHECK(error == "r: storage buffer data needs WebGPU; WebGL2 cannot run it");
}

TEST_CASE("reflection lists feature uses that backends compare with their capabilities") {
  const ProgramAST ast = computeProgram("k", "K", {1, 1, 1}, [] {
    const StorageBuffer data = storage("data", u32(), StorageAccess::ReadWrite);
    data[0u].set(1u);
    workgroupBarrier();
  });
  const Reflection r = analyzeProgram(ast).reflection;
  REQUIRE(r.features.size() == 3);
  CHECK(r.features[0] == FeatureUse{Feature::ComputeProgram, "a compute program"});
  CHECK(r.features[1] == FeatureUse{Feature::StorageBuffer, "storage buffer data"});
  CHECK(r.features[2] == FeatureUse{Feature::Barrier, "workgroupBarrier"});
  CHECK(unsupported(r.features, webgpuCapabilities()).empty());
  // WebGL2 lacks all three
  CHECK(unsupported(r.features, webgl2Capabilities(true)) == r.features);

  const std::vector<FeatureUse> flat{{Feature::FlatFirstVertex, "flat varying id"}};
  CHECK(unsupported(flat, webgl2Capabilities(false)) == flat);
  CHECK(unsupported(flat, webgl2Capabilities(true)).empty());
}

TEST_CASE("fragment stage storage writes are valid, vertex stage ones are not") {
  const StorageBuffer hits = storage("hits", u32());
  const auto [wgsl, error] = renderBoth([&](const Output& out) {
    hits[toU32(fragCoord().x())].set(1u);
    out.set(vec4(1.0));
  });
  CHECK(has(wgsl, "sb_hits[u32(fin.fragCoord.x)] = 1u;"));
  CHECK_FALSE(error.empty());

  CHECK_THROWS_AS(program("v", "V", [&] {
    hits[0u].set(1u);
    Position(vec4(0.0));
  }, [] {}),
                  ShaderError);
  // a read/write buffer cannot even be read in the vertex stage
  const ProgramAST readsRW = program("v", "V", [&] { Position(vec4(toF32(hits[0u]))); }, [] {});
  CHECK_THROWS_AS(analyzeProgram(readsRW), ShaderError);
  // a read only buffer can
  const StorageBuffer weights = storage("weights", f32(), StorageAccess::Read);
  const ProgramAST readsRO = program("v", "V", [&] { Position(vec4(weights[0u])); }, [] {});
  CHECK_NOTHROW(analyzeProgram(readsRO));
}

TEST_CASE("atomics are checked") {
  CHECK_THROWS_AS(storage("w", f32(), StorageAccess::Write), ShaderError);
  // a read only buffer cannot hold atomics at all
  CHECK_THROWS_AS(storage("ro", atomicU32(), StorageAccess::Read), ShaderError);
  const StorageBuffer rw = storage("rw", atomicU32());
  const StorageBuffer plain = storage("plain", u32());
  computeProgram("a", "A", {1, 1, 1}, [&] {
    CHECK_THROWS_AS(atomicAdd(plain[0u], 1u), ShaderError);
    CHECK_THROWS_AS(rw[0u].set(1u), ShaderError);
    CHECK_THROWS_AS(atomicAdd(rw[0u], 1.5), ShaderError);
    CHECK(atomicExchange(rw[0u], 3u).type() == u32());
  });
  CHECK_THROWS_AS(structLayout(makeStruct("U", {{"n", atomicU32()}})), ShaderError);
}

TEST_CASE("compute only features are rejected elsewhere") {
  const Val local = workgroupArray("local", f32(), 4);
  CHECK_THROWS_AS(renderBoth([](const Output&) { workgroupBarrier(); }), ShaderError);
  CHECK_THROWS_AS(renderBoth([](const Output&) { (void)globalId(); }), ShaderError);
  CHECK_THROWS_AS(renderBoth([&](const Output& out) { out.set(vec4(local[0])); }), ShaderError);
  CHECK_THROWS_AS(computeProgram("z", "Z", {0, 1, 1}, [] {}), ShaderError);
}

TEST_CASE("storage textures") {
  const StorageTexture image =
      storageTexture("image", TextureDimension::D2, TexelFormat::Rgba16Float);
  const StorageTexture counts = storageTexture("counts", TextureDimension::D2Array,
                                               TexelFormat::R32Uint, StorageAccess::ReadWrite);
  const ProgramAST ast = computeProgram("s", "S", {8, 8, 1}, [&] {
    const Val p = Let(globalId().xy(), "p");
    const Val c = Let(counts.load(p, 0), "c");
    counts.store(p, c + 1u, 0);
    image.store(p, vec4(toF32(c.x()), 0.0, 0.0, 1.0));
    const Val size = Let(counts.size(), "size");
    CHECK(size.type() == vec3u());
  });
  const std::string w = compileWGSL(analyzeProgram(ast)).code;
  CHECK(has(w, "var st_image: texture_storage_2d<rgba16float, write>;"));
  CHECK(has(w, "var st_counts: texture_storage_2d_array<r32uint, read_write>;"));
  CHECK(has(w, "textureLoad(st_counts, p_"));
  CHECK(has(w, "textureStore(st_image, p_"));
  CHECK(has(w, "vec3u(textureDimensions(st_counts), textureNumLayers(st_counts))"));

  // rgba8snorm has no read_write access, even with texture-formats-tier2
  CHECK_THROWS_WITH_AS(storageTexture("rw", TextureDimension::D2, TexelFormat::Rgba8Snorm, StorageAccess::ReadWrite),
                       doctest::Contains("Storage texture rw: WebGPU has no read/write rgba8snorm storage texture"), ShaderError);
  CHECK_THROWS_AS(storageTexture("cube", TextureDimension::Cube, TexelFormat::R32Float), ShaderError);
  const StorageTexture readOnly = storageTexture("ro", TextureDimension::D2, TexelFormat::R32Float,
                                                 StorageAccess::Read);
  computeProgram("e", "E", {1, 1, 1}, [&] {
    CHECK_THROWS_AS(readOnly.store(uvec2(0, 0), vec4(0.0)), ShaderError);
    CHECK_THROWS_AS(image.load(uvec2(0, 0)), ShaderError);
    CHECK_THROWS_AS(counts.store(uvec2(0, 0), uvec4(0)), ShaderError);
    CHECK_THROWS_AS(image.store(vec2(0.0), vec4(0.0)), ShaderError);
  });
}

TEST_CASE("subgroups enable the extension and check their arguments") {
  const StorageBuffer out = storage("out", f32());
  const ProgramAST ast = computeProgram("g", "G", {32, 1, 1}, [&] {
    const Val v = Let(toF32(subgroupInvocationId()), "v");
    const Val s = Let(subgroupAdd(v) + subgroupShuffle(v, localIndex() ^ 1u) + subgroupBroadcast(v, 0),
                      "s");
    If(subgroupElect(), [&] { out[workgroupId().x()].set(s); });
    CHECK(subgroupBallot(v > 0.0).type() == vec4u());
    CHECK_THROWS_AS(subgroupBroadcast(v, localIndex()), ShaderError);
    CHECK_THROWS_AS(subgroupAnd(v), ShaderError);
  });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(a.reflection.deviceFeatures == std::vector{DeviceFeature::Subgroups});
  const std::string w = compileWGSL(a).code;
  CHECK(w.starts_with("enable subgroups;"));
  CHECK(has(w, "@builtin(subgroup_invocation_id) subgroupInvocationId: u32,"));
  CHECK(has(w, "subgroupBroadcast(l_v_"));
  CHECK(has(w, ", 0u)"));
  CHECK_THROWS_AS(program("v", "V", [] { (void)subgroupSize(); }, [] {}), ShaderError);
}

TEST_CASE("a bgra8unorm storage texture needs the bgra8unorm-storage device feature") {
  const StorageTexture bgra = storageTexture("bgra", TextureDimension::D2, TexelFormat::Bgra8Unorm);
  const StorageTexture rgba = storageTexture("rgba", TextureDimension::D2, TexelFormat::Rgba8Unorm);
  const auto run = [](const StorageTexture& t) {
    return analyzeProgram(computeProgram("b", "B", {8, 8, 1}, [&] {
             t.store(globalId().xy(), vec4(1.0));
           })).reflection;
  };
  const Reflection withBgra = run(bgra);
  CHECK(withBgra.deviceFeatures == std::vector{DeviceFeature::Bgra8UnormStorage});
  CHECK(withBgra.needs(DeviceFeature::Bgra8UnormStorage));
  // the name WebGPU uses in GPUDeviceDescriptor.requiredFeatures
  CHECK(deviceFeatureName(DeviceFeature::Bgra8UnormStorage) == "bgra8unorm-storage");
  CHECK(deviceFeatureName(DeviceFeature::Subgroups) == "subgroups");
  // rgba8unorm is a core storage format
  CHECK(run(rgba).deviceFeatures.empty());
}

namespace {

/** The names in the enable directives of WGSL code, in order. */
std::vector<std::string> enables(const std::string& wgsl) {
  std::vector<std::string> names;
  std::size_t at = 0;
  while ((at = wgsl.find("enable ", at)) != std::string::npos) {
    const std::size_t end = wgsl.find(';', at);
    names.push_back(wgsl.substr(at + 7, end - at - 7));
    at = end;
  }
  return names;
}

} // namespace

TEST_CASE("WGSL enable directives match the reported device features") {
  const Output out = output(0);
  const StorageTexture bgra = storageTexture("bgra", TextureDimension::D2, TexelFormat::Bgra8Unorm);
  // the WGSL extension of each feature, from the WGSL specification
  const ProgramAST clip = program("c", "C", [] {
    ClipDistance(0, 1.0);
    Position(vec4(0.0));
  }, [&] { out.set(vec4(1.0)); });
  const ProgramAST sub = program("s", "S", [] { Position(vec4(0.0)); },
                                 [&] { out.set(vec4(subgroupAdd(fragCoord().x()))); });
  const ProgramAST both = program("b", "B", [] {
    ClipDistance(0, 1.0);
    Position(vec4(0.0));
  }, [&] { out.set(vec4(subgroupAdd(fragCoord().x()))); });
  const ProgramAST storage = computeProgram("t", "T", {8, 8, 1}, [&] { bgra.store(globalId().xy(), vec4(1.0)); });
  const ProgramAST plain = program("p", "P", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); });

  const auto check = [](const ProgramAST& ast, const std::vector<DeviceFeature>& features,
                        const std::vector<std::string>& expected) {
    const AnalyzedProgram a = analyzeProgram(ast);
    CHECK(a.reflection.deviceFeatures == features);
    CHECK(enables(compileWGSL(a).code) == expected);
  };
  check(clip, {DeviceFeature::ClipDistances}, {"clip_distances"});
  check(sub, {DeviceFeature::Subgroups}, {"subgroups"});
  check(both, {DeviceFeature::ClipDistances, DeviceFeature::Subgroups}, {"clip_distances", "subgroups"});
  // bgra8unorm-storage is a device feature without a WGSL directive
  check(storage, {DeviceFeature::Bgra8UnormStorage}, {});
  check(plain, {}, {});
}

TEST_CASE("primitiveIndex is a WebGPU fragment builtin that needs primitive-index") {
  const Output out = output(0, vec4u());
  const ProgramAST ast = program("p", "P", [] { Position(vec4(0.0)); },
                                 [&] { out.set(uvec4(primitiveIndex())); });
  const AnalyzedProgram a = analyzeProgram(ast);
  CHECK(a.reflection.deviceFeatures == std::vector{DeviceFeature::PrimitiveIndex});
  CHECK(deviceFeatureName(DeviceFeature::PrimitiveIndex) == "primitive-index");
  const std::string w = compileWGSL(a).code;
  CHECK(enables(w) == std::vector<std::string>{"primitive_index"});
  CHECK(has(w, "@builtin(primitive_index) primitiveIndex: u32,"));
  // GLSL ES 3.00 has no gl_PrimitiveID
  CHECK_THROWS_WITH_AS(compileGLSL(a), doctest::Contains("p: builtin primitiveIndex needs WebGPU; WebGL2 cannot run it"), ShaderError);
  CHECK_THROWS_AS(program("v", "V", [] { Position(vec4(toF32(primitiveIndex()))); }, [] {}), ShaderError);
}

TEST_CASE("storage texture formats report the texture format tiers they need") {
  using DF = DeviceFeature;
  const std::vector<DF> tier1{DF::Rg11b10UfloatRenderable, DF::TextureFormatsTier1};
  const std::vector<DF> tier2{DF::Rg11b10UfloatRenderable, DF::TextureFormatsTier1, DF::TextureFormatsTier2};
  // expected values measured on Chrome 155: GPUDevice.createBindGroupLayout with each feature or not
  const auto needs = [](TexelFormat f, StorageAccess a) { return storageFormatFeatures(f, a); };
  CHECK(needs(TexelFormat::Rgba8Unorm, StorageAccess::Write) == std::vector<DF>{});
  CHECK(needs(TexelFormat::R32Float, StorageAccess::ReadWrite) == std::vector<DF>{});
  CHECK(needs(TexelFormat::Rgba8Unorm, StorageAccess::ReadWrite) == tier2);
  CHECK(needs(TexelFormat::R8Unorm, StorageAccess::Write) == tier1);
  CHECK(needs(TexelFormat::R8Unorm, StorageAccess::Read) == tier1);
  CHECK(needs(TexelFormat::R8Unorm, StorageAccess::ReadWrite) == tier2);
  CHECK(needs(TexelFormat::Rg11b10Ufloat, StorageAccess::Write) == tier1);
  CHECK(needs(TexelFormat::Bgra8Unorm, StorageAccess::Write) == std::vector{DF::Bgra8UnormStorage});
  CHECK_FALSE(needs(TexelFormat::Bgra8Unorm, StorageAccess::Read).has_value());
  CHECK_FALSE(needs(TexelFormat::Rg16Float, StorageAccess::ReadWrite).has_value());
  CHECK_FALSE(needs(TexelFormat::R8Snorm, StorageAccess::ReadWrite).has_value());
  CHECK_THROWS_AS(storageTexture("b", TextureDimension::D2, TexelFormat::Bgra8Unorm, StorageAccess::Read), ShaderError);

  // a program that uses the texture reports the tier, ready for requiredFeatures
  const StorageTexture rw = storageTexture("rw", TextureDimension::D2, TexelFormat::Rgba8Unorm, StorageAccess::ReadWrite);
  const StorageTexture small = storageTexture("small", TextureDimension::D2, TexelFormat::R16Unorm);
  const AnalyzedProgram a = analyzeProgram(computeProgram("t", "T", {8, 8, 1}, [&] {
    rw.store(globalId().xy(), rw.load(globalId().xy()));
    small.store(globalId().xy(), vec4(1.0));
  }));
  CHECK(a.reflection.deviceFeatures == tier2);
  std::vector<std::string_view> names;
  for (const DF f : a.reflection.deviceFeatures) names.push_back(deviceFeatureName(f));
  CHECK(names == std::vector<std::string_view>{"rg11b10ufloat-renderable", "texture-formats-tier1", "texture-formats-tier2"});
  const std::string w = compileWGSL(a).code;
  CHECK(has(w, "var st_rw: texture_storage_2d<rgba8unorm, read_write>;"));
  CHECK(has(w, "var st_small: texture_storage_2d<r16unorm, write>;"));
  // the tiers are WGSL language features: no enable directive
  CHECK(enables(w).empty());
}

TEST_CASE("a compute program can fix its subgroup size") {
  const AnalyzedProgram a = analyzeProgram(computeProgram("s", "S", {{64, 1, 1}, 32}, [] {}));
  CHECK(a.reflection.subgroupSize == 32u);
  // subgroup-size-control needs subgroups on the device
  CHECK(a.reflection.deviceFeatures == std::vector{DeviceFeature::Subgroups, DeviceFeature::SubgroupSizeControl});
  CHECK(deviceFeatureName(DeviceFeature::SubgroupSizeControl) == "subgroup-size-control");
  const std::string w = compileWGSL(a).code;
  CHECK(has(w, "@compute @workgroup_size(64, 1, 1) @subgroup_size(32) fn cs_main("));
  // Tint accepts the directive pair (Chrome 155): "@subgroup_size requires enabling extension subgroup_size_control"
  CHECK(enables(w) == std::vector<std::string>{"subgroups", "subgroup_size_control"});
  CHECK_THROWS_WITH_AS(computeProgram("b", "B", {{64, 1, 1}, 3}, [] {}), doctest::Contains("b: the subgroup size 3 is not a power of 2"),
                       ShaderError);
  CHECK_FALSE(analyzeProgram(computeProgram("n", "N", {8, 8, 1}, [] {})).reflection.subgroupSize.has_value());
  // Chrome 155: "The x-dimension of workgroup invocations (8) is not a multiple of the subgroup_size attribute (32)"
  CHECK_THROWS_WITH_AS(computeProgram("x", "X", {{8, 8, 1}, 32}, [] {}),
                       doctest::Contains("x: the workgroup x size 8 is not a multiple of the subgroup size 32"), ShaderError);
  CHECK_NOTHROW(computeProgram("y", "Y", {{32, 2, 1}, 32}, [] {}));
}

TEST_CASE("retained resources report the device features their bindings need") {
  const StorageTexture bgra = storageTexture("bgra", TextureDimension::D2, TexelFormat::Bgra8Unorm);
  const Texture field = texture("field", TextureDimension::D2, {.float32Filterable = true});
  const ProgramAST ast = computeProgram("r", "R", {8, 8, 1}, [] {});
  const AnalyzedProgram a = analyzeProgram(ast, {.retain = {bgra.resource(), field.resource()}});
  // WebGPU checks storage texture formats and filterable samplers when it creates the bind group layout
  std::vector<DeviceFeature> features = a.reflection.deviceFeatures;
  std::ranges::sort(features);
  CHECK(features == std::vector{DeviceFeature::Bgra8UnormStorage, DeviceFeature::Float32Filterable});
  CHECK(has(compileWGSL(a).code, "var st_bgra: texture_storage_2d<bgra8unorm, write>;"));

  // a retained storage texture alone still makes the program WebGPU only
  const Output out = output(0);
  const ProgramAST render = program("v", "V", [] { Position(vec4(0.0)); }, [&] { out.set(vec4(1.0)); });
  const AnalyzedProgram b = analyzeProgram(render, {.retain = {bgra.resource()}});
  CHECK_THROWS_WITH_AS(compileGLSL(b), doctest::Contains("v: storage texture bgra needs WebGPU; WebGL2 cannot run it"), ShaderError);
}

TEST_CASE("read and read_write storage textures, textureBarrier and tier 1 formats need WGSL language features") {
  using LF = LanguageFeature;
  CHECK(languageFeatureName(LF::ReadonlyAndReadwriteStorageTextures) == "readonly_and_readwrite_storage_textures");
  CHECK(languageFeatureName(LF::TextureFormatsTier1) == "texture_formats_tier1");
  const auto analyze = [](const std::function<void()>& body) {
    return analyzeProgram(computeProgram("l", "L", {8, 8, 1}, body));
  };
  const StorageTexture writeOnly = storageTexture("w", TextureDimension::D2, TexelFormat::Rgba8Unorm);
  const StorageTexture readOnly = storageTexture("ro", TextureDimension::D2, TexelFormat::R32Float, StorageAccess::Read);
  const StorageTexture readWrite = storageTexture("rw", TextureDimension::D2, TexelFormat::Rgba8Unorm, StorageAccess::ReadWrite);
  const StorageTexture tier1 = storageTexture("t1", TextureDimension::D2, TexelFormat::R8Unorm);

  CHECK(analyze([&] { writeOnly.store(globalId().xy(), vec4(1.0)); }).reflection.languageFeatures.empty());
  CHECK(analyze([&] { writeOnly.store(globalId().xy(), readOnly.load(globalId().xy())); }).reflection.languageFeatures ==
        std::vector{LF::ReadonlyAndReadwriteStorageTextures});
  // rgba8unorm is a core format: read_write needs tier2 on the device, but no tier 1 format in WGSL
  CHECK(analyze([&] { readWrite.store(globalId().xy(), readWrite.load(globalId().xy())); }).reflection.languageFeatures ==
        std::vector{LF::ReadonlyAndReadwriteStorageTextures});
  CHECK(analyze([] { textureBarrier(); }).reflection.languageFeatures == std::vector{LF::ReadonlyAndReadwriteStorageTextures});
  const AnalyzedProgram t = analyze([&] { tier1.store(globalId().xy(), vec4(1.0)); });
  CHECK(t.reflection.languageFeatures == std::vector{LF::TextureFormatsTier1});
  CHECK(has(compileWGSL(t).code, "requires texture_formats_tier1;"));
  const AnalyzedProgram both = analyze([&] {
    tier1.store(globalId().xy(), readOnly.load(globalId().xy()));
  });
  CHECK(has(compileWGSL(both).code, "requires readonly_and_readwrite_storage_textures, texture_formats_tier1;"));
}

TEST_CASE("a retained storage buffer declares its element struct") {
  const TypePtr particle = makeStruct("Particle", {{"position", vec4f()}, {"life", f32()}});
  const StorageBuffer particles = storage("particles", particle);
  const ProgramAST ast = computeProgram("r", "R", {8, 8, 1}, [] {});
  const AnalyzedProgram a = analyzeProgram(ast, {.retain = {particles.resource()}});
  const std::string w = compileWGSL(a).code;
  CHECK(has(w, "var<storage, read_write> sb_particles: array<S_Particle>;"));
  // the storage form: no explicit uniform @align/@size attributes; without it Tint reports
  // "unresolved type 'S_Particle'" (Chrome 155)
  CHECK(has(w, "struct S_Particle {\n  position: vec4f,\n  life: f32,\n}"));
}

TEST_CASE("an f16 storage buffer needs shader-f16 when nothing reads its elements") {
  const StorageBuffer halves = storage("halves", vec4h());
  // only arrayLength() reads the buffer: the declaration still names vec4h
  const StorageBuffer counts = storage("counts", u32());
  const AnalyzedProgram used = analyzeProgram(computeProgram("l", "L", {8, 8, 1}, [&] {
    counts[0u].set(halves.length());
  }));
  CHECK(used.reflection.deviceFeatures == std::vector{DeviceFeature::ShaderF16});
  const std::string w = compileWGSL(used).code;
  CHECK(has(w, "enable f16;"));
  CHECK(has(w, "array<vec4h>"));
  // retained and never used
  const AnalyzedProgram retained = analyzeProgram(computeProgram("r", "R", {8, 8, 1}, [] {}), {.retain = {halves.resource()}});
  CHECK(retained.reflection.deviceFeatures == std::vector{DeviceFeature::ShaderF16});
  CHECK(has(compileWGSL(retained).code, "enable f16;"));
}

TEST_CASE("a storage buffer rejects a known negative index") {
  const StorageBuffer data = storage("data", u32());
  const Val back = constant("BACK", toI32(0) - toI32(1));
  computeProgram("n", "N", {8, 1, 1}, [&] {
    CHECK_THROWS_WITH_AS(data[toI32(-1)], doctest::Contains("data: index -1 is negative"), ShaderError);
    CHECK_THROWS_AS(data[back], ShaderError);
    // the length is the bound buffer's, so a large index and a run time index stay valid
    CHECK_NOTHROW(data[toI32(1000000)]);
    CHECK_NOTHROW(data[toI32(localIndex()) - toI32(1)]);
    data[0u].set(1u);
  });
}

TEST_CASE("barriers must be in uniform control flow") {
  // each case was compiled by Tint (Chrome 155); `true` means Tint rejected it
  const Val gate = uniform("gate", u32(), ResourceFrequency::Frame);
  const StorageBuffer rw = storage("rw", u32());
  const StorageBuffer ro = storage("ro", u32(), StorageAccess::Read);
  const Val limit = constant("LIMIT", toU32(3));
  const auto check = [](bool rejected, const std::function<void()>& body) {
    const auto analyze = [&] { analyzeProgram(computeProgram("u", "U", {64, 1, 1}, body)); };
    if (rejected) CHECK_THROWS_WITH_AS(analyze(), doctest::Contains("must be reached by all invocations together"), ShaderError);
    else CHECK_NOTHROW(analyze());
  };
  const auto barrierIf = [](const Val& c) { If(c, [] { workgroupBarrier(); }); };
  check(true, [&] { barrierIf(localId().x().eq(0u)); });
  check(true, [&] { barrierIf(globalId().x().eq(0u)); });
  check(false, [&] { barrierIf(workgroupId().x().eq(0u)); });
  check(false, [&] { barrierIf(numWorkgroups().x().eq(1u)); });
  check(false, [&] { barrierIf(gate.eq(0u)); });
  check(false, [&] { barrierIf(limit.eq(3u)); });
  check(true, [&] { barrierIf(rw[0u].eq(0u)); });
  check(false, [&] { barrierIf(ro[0u].eq(0u)); });
  check(true, [&] { barrierIf(ro[localId().x()].eq(0u)); });
  check(true, [&] { barrierIf(workgroupArray("wg", u32(), 4)[0].eq(0u)); });
  check(true, [&] { barrierIf(select(localId().x().eq(0u), 1u, 0u).eq(0u)); });
  // a local assigned under a non uniform condition is non uniform
  check(true, [&] {
    const Val c = Var(toU32(0), "c");
    If(localId().x().eq(0u), [&] { c.set(1u); });
    barrierIf(c.eq(0u));
  });
  check(false, [&] {
    const Val c = Var(toU32(0), "c");
    If(localId().x().eq(0u), [&] { c.set(1u); });
    workgroupBarrier();
    rw[0u].set(c);
  });
  check(false, [&] {
    const Val c = Var(toU32(0), "c");
    If(gate.eq(0u), [&] { c.set(1u); });
    barrierIf(c.eq(0u));
  });
  // control flow joins again after an if or a loop, but not after an uneven return, break or continue
  check(false, [&] {
    If(localId().x().eq(0u), [&] { rw[0u].set(1u); });
    workgroupBarrier();
  });
  check(true, [&] {
    If(localId().x().eq(0u), [] { Return(); });
    workgroupBarrier();
  });
  check(true, [&] {
    For(toU32(0), 4u, [](const Val& i) {
      If(localId().x().eq(i), [] { Break(); });
      workgroupBarrier();
    });
  });
  check(false, [&] {
    For(toU32(0), 4u, [](const Val& i) { If(localId().x().eq(i), [] { Break(); }); });
    workgroupBarrier();
  });
  check(true, [&] {
    For(toU32(0), 4u, [](const Val& i) {
      If(localId().x().eq(i), [] { Continue(); });
      workgroupBarrier();
    });
  });
  check(false, [&] { For(toU32(0), gate, [](const Val&) { workgroupBarrier(); }); });
  check(true, [&] { For(toU32(0), localId().x(), [](const Val&) { workgroupBarrier(); }); });
  check(true, [&] { Switch(localId().x()).Case(0u, [] { workgroupBarrier(); }); });
  // functions: a barrier inside, and the uniformity of a result
  check(true, [&] {
    const Function g = fn("g", {}, voidType(), [](const Args&) { workgroupBarrier(); });
    If(localId().x().eq(0u), [&] { g(); });
  });
  check(false, [&] {
    const Function h = fn("h", {{"x", u32()}}, u32(), [](const Args& a) { Return(a["x"] * 2u); });
    barrierIf(h(workgroupId().x()).eq(0u));
  });
  check(true, [&] {
    const Function h2 = fn("h2", {}, u32(), [&](const Args&) { Return(rw[0u]); });
    barrierIf(h2().eq(0u));
  });
  // which return a helper takes decides its result: uniform values, uneven branch
  check(true, [&] {
    const Function h3 = fn("h3", {}, u32(), [&](const Args&) {
      If(rw[0u].eq(0u), [] { Return(1u); });
      Return(0u);
    });
    barrierIf(h3().eq(0u));
  });
  check(false, [&] {
    const Function h4 = fn("h4", {}, u32(), [&](const Args&) {
      If(gate.eq(0u), [] { Return(1u); });
      Return(0u);
    });
    barrierIf(h4().eq(0u));
  });
  // a loop header runs again for the invocations that did not return from the body
  check(true, [&] {
    const Function g = fn("g", {}, u32(), [](const Args&) {
      workgroupBarrier();
      Return(4u);
    });
    For(toU32(0), g(), [&](const Val& i) { If(localId().x().eq(i), [] { Return(); }); });
  });
  check(false, [&] {
    const Function g = fn("g2", {}, u32(), [](const Args&) {
      workgroupBarrier();
      Return(4u);
    });
    For(toU32(0), g(), [&](const Val& i) { If(gate.eq(i), [] { Return(); }); });
  });
  // the subgroup size is uniform in a compute dispatch
  check(false, [&] { barrierIf(subgroupSize().eq(32u)); });
  // a subgroup result differs between invocations even for a uniform argument
  check(true, [&] { barrierIf(subgroupAdd(gate) > 0u); });
  // a helper's result depends only on the arguments that reach a return
  check(false, [&] {
    const Function zero = fn("zero", {{"x", u32()}}, u32(), [](const Args&) { Return(0u); });
    barrierIf(zero(localId().x()).eq(0u));
  });
  check(true, [&] {
    const Function same = fn("same", {{"x", u32()}}, u32(), [](const Args& a) { Return(a["x"]); });
    barrierIf(same(localId().x()).eq(0u));
  });
  check(true, [&] {
    const Function pick = fn("pick", {{"x", u32()}}, u32(), [](const Args& a) {
      If(a["x"].eq(0u), [] { Return(1u); });
      Return(0u);
    });
    barrierIf(pick(localId().x()).eq(0u));
  });
  // subgroup operations combine invocations, like barriers (subgroup_uniformity is an error by default)
  const auto subgroupIf = [&](const Val& c, const std::function<Val()>& op) { If(c, [&] { rw[0u].set(op()); }); };
  check(true, [&] { subgroupIf(localId().x().eq(0u), [] { return subgroupAdd(localId().x()); }); });
  check(true, [&] { subgroupIf(localId().x().eq(0u), [] { return toU32(subgroupElect()); }); });
  check(true, [&] { subgroupIf(localId().x().eq(0u), [] { return subgroupBallot(toBool(1u)).x(); }); });
  check(true, [&] { subgroupIf(localId().x().eq(0u), [] { return subgroupBroadcast(localId().x(), 0u); }); });
  check(true, [&] { subgroupIf(localId().x().eq(0u), [] { return subgroupShuffle(localId().x(), 0u); }); });
  check(false, [&] { subgroupIf(gate.eq(0u), [] { return subgroupAdd(localId().x()); }); });
  // continue in a switch case belongs to the loop; break only ends the case
  check(true, [&] {
    For(toU32(0), 4u, [](const Val&) {
      Switch(localId().x()).Case(0u, [] { Continue(); });
      workgroupBarrier();
    });
  });
  check(false, [&] {
    For(toU32(0), 4u, [](const Val&) {
      Switch(localId().x()).Case(0u, [] { Break(); });
      workgroupBarrier();
    });
  });
}

TEST_CASE("derivatives must be in uniform control flow") {
  const Val gate = uniform("fade", f32(), ResourceFrequency::Frame);
  const Varying uv = varying("uv", vec2f());
  const Varying id = varying("id", u32(), Interpolation::Flat);
  const Output out = output(0);
  // `true` means Tint (Chrome 155) rejected the equivalent WGSL
  const auto check = [&](bool rejected, const std::function<void()>& fragment) {
    const auto analyze = [&] {
      analyzeProgram(program("d", "D", [&] {
        uv.set(vec2(0.0));
        id.set(0u);
        Position(vec4(0.0));
      }, fragment));
    };
    if (rejected) CHECK_THROWS_WITH_AS(analyze(), doctest::Contains("must be reached by all invocations together"), ShaderError);
    else CHECK_NOTHROW(analyze());
  };
  const auto derivativeIf = [&](const Val& c) { If(c, [&] { out.set(vec4(dpdx(uv.value().x()))); }); };
  check(true, [&] { derivativeIf(fragCoord().x() > 1.0); });
  check(true, [&] { derivativeIf(uv.value().x() > 1.0); });
  check(true, [&] { derivativeIf(id.value().eq(0u)); });
  check(false, [&] { derivativeIf(gate > 1.0); });
  check(false, [&] {
    If(uv.value().x() > 1.0, [] { Discard(); });
    out.set(vec4(dpdx(uv.value().x())));
  });
  check(true, [&] {
    If(uv.value().x() > 1.0, [&] {
      out.set(vec4(1.0));
      Return();
    });
    out.set(vec4(dpdx(uv.value().x())));
  });
  // the right operand of && and || runs only where the left one lets it
  // a loop header runs again after each pass, for only the invocations still looping
  check(true, [&] {
    For(ForRange{0, toI32(dpdx(uv.value().x())), 1, BinaryOp::Lt}, [&](const Val&) {
      If(uv.value().y() > 0.0, [] { Break(); });
    });
    out.set(vec4(1.0));
  });
  check(true, [&] {
    const Val i = Var(0.0, "i");
    While(dpdx(uv.value().x()) > i, [&] { i.addAssign(1.0); });
    out.set(vec4(i));
  });
  check(true, [&] { If(uv.value().x() > 0.0, [&] { out.set(vec4(subgroupAdd(uv.value().x()))); }); });
  // a derivative differs between fragments even for a uniform argument
  check(true, [&] { derivativeIf(dpdx(gate) > 0.0); });
  // the subgroup size is not uniform across the fragments of a draw
  check(true, [&] { derivativeIf(toF32(subgroupSize()) > 1.0); });
  // (as in the WGSL checked in Chrome, the body itself has no derivative)
  const auto plainIf = [&](const Val& c) { If(c, [&] { out.set(vec4(1.0)); }); };
  check(true, [&] { plainIf((uv.value().x() > 1.0) && (dpdx(uv.value().x()) > 0.0)); });
  check(true, [&] { plainIf((uv.value().x() > 1.0) || (dpdx(uv.value().x()) > 0.0)); });
  check(false, [&] { plainIf((gate > 1.0) && (dpdx(uv.value().x()) > 0.0)); });
}

TEST_CASE("a shared read of writable memory stays after the calls of its statement") {
  const StorageBuffer rw = storage("cells", f32());
  const AnalyzedProgram a = analyzeProgram(computeProgram("o", "O", {8, 1, 1}, [&] {
    const Function bump = fn("bump", {}, f32(), [&](const Args&) {
      rw[0u].set(1.0);
      Return(1.0);
    });
    // a large expression reusing a storage read, next to a call that writes it
    Val r = rw[0u] * 1.0;
    for (int i = 0; i < 8; ++i) r = r + r;
    rw[1u].set(select(bump() > 0.0, r, 0.0));
    // without a call in the statement the reuse is still printed once
    rw[2u].set(r);
  }));
  const std::string w = compileWGSL(a).code;
  INFO(w);
  // the call's statement keeps its read of sb_cells[0] in place: no local holds it
  std::size_t hoistedReads = 0;
  for (std::size_t at = w.find("let moga_shared_"); at != std::string::npos; at = w.find("let moga_shared_", at + 1)) {
    const std::string line = w.substr(at, w.find('\n', at) - at);
    if (line.find("sb_cells[0u]") != std::string::npos) ++hoistedReads;
  }
  // the second statement, which has no call, holds the read in one local
  CHECK(hoistedReads == 1);
  CHECK(w.find("f_bump()") < w.find("let moga_shared_"));
}

TEST_CASE("a user local named like a generated one keeps its own name") {
  const AnalyzedProgram a = analyzeProgram(computeProgram("n", "N", {8, 1, 1}, [&] {
    const StorageBuffer out = storage("outs", f32());
    const Val clash = Var(toF32(localIndex()), "moga_shared");
    Val r = clash * 1.0;
    for (int i = 0; i < 8; ++i) r = r + r;
    out[0u].set(r);
  }));
  const std::string w = compileWGSL(a).code;
  // generated locals start with moga_; a user local never does
  CHECK(has(w, "var l_moga_shared_"));
  CHECK(has(w, "let moga_shared_0"));
}

TEST_CASE("the minimum i32 literal is valid WGSL") {
  const StorageBuffer out = storage("mins", i32());
  const AnalyzedProgram a = analyzeProgram(computeProgram("m", "M", {8, 1, 1}, [&] {
    out[0u].set(toI32(-2147483648.0));
    out[1u].set(toI32(-5));
  }));
  const std::string w = compileWGSL(a).code;
  // Tint (Chrome 155): (-2147483648i) "cannot be represented as i32"; i32(-2147483648) is valid
  CHECK(has(w, "= i32(-2147483648);"));
  CHECK_FALSE(has(w, "2147483648i"));
  CHECK(has(w, "= (-5i);"));
}

TEST_CASE("a struct may be in a uniform block and a storage buffer when the layouts match") {
  // one vec4f: offset 0, size 16, align 16 in std140 and in WGSL storage
  const TypePtr same = makeStruct("Same", {{"v", vec4f()}});
  const UniformBlock ub = uniformBlock("u", {{"s", same}});
  const ProgramAST ok = computeProgram("k", "K", {1, 1, 1}, [&] {
    const StorageBuffer data = storage("data", same, StorageAccess::ReadWrite);
    data[0u].set(ub["s"]);
  });
  const std::string wgsl = compileWGSL(analyzeProgram(ok)).code;
  CHECK(has(wgsl, "struct S_Same {"));

  // two f32: std140 rounds the struct size up to 16, WGSL storage keeps 8
  const TypePtr pair = makeStruct("Pair", {{"a", f32()}, {"b", f32()}});
  const UniformBlock ub2 = uniformBlock("u2", {{"p", pair}});
  const ProgramAST bad = computeProgram("k", "K", {1, 1, 1}, [&] {
    const StorageBuffer data = storage("data", pair, StorageAccess::ReadWrite);
    data[0u].set(ub2["p"]);
  });
  CHECK_THROWS_WITH_AS(analyzeProgram(bad), doctest::Contains("Struct Pair is used in a uniform block and in a storage buffer"),
                       ShaderError);
}

TEST_CASE("an out of range texel format is rejected") {
  CHECK_THROWS_AS(storageFormatFeatures(static_cast<TexelFormat>(40), StorageAccess::Read), ShaderError);
  CHECK_THROWS_AS(storageFormatFeatures(static_cast<TexelFormat>(1000), StorageAccess::Read), ShaderError);
  CHECK_THROWS_AS(texelFormatName(static_cast<TexelFormat>(40)), ShaderError);
  CHECK_THROWS_AS(texelFormatScalar(static_cast<TexelFormat>(40)), ShaderError);
}
