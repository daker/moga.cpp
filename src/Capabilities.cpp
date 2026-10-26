#include "moga/Capabilities.hpp"

#include <array>
#include <format>

namespace moga {

std::string_view deviceFeatureName(DeviceFeature f) {
  switch (f) {
  case DeviceFeature::Subgroups:
    return "subgroups";
  case DeviceFeature::Bgra8UnormStorage:
    return "bgra8unorm-storage";
  case DeviceFeature::ClipDistances:
    return "clip-distances";
  case DeviceFeature::Float32Filterable:
    return "float32-filterable";
  case DeviceFeature::PrimitiveIndex:
    return "primitive-index";
  case DeviceFeature::Rg11b10UfloatRenderable:
    return "rg11b10ufloat-renderable";
  case DeviceFeature::TextureFormatsTier1:
    return "texture-formats-tier1";
  case DeviceFeature::TextureFormatsTier2:
    return "texture-formats-tier2";
  case DeviceFeature::SubgroupSizeControl:
    return "subgroup-size-control";
  case DeviceFeature::DualSourceBlending:
    return "dual-source-blending";
  case DeviceFeature::ShaderF16:
    return "shader-f16";
  }
  return "?";
}

std::string_view languageFeatureName(LanguageFeature f) {
  switch (f) {
  case LanguageFeature::ReadonlyAndReadwriteStorageTextures:
    return "readonly_and_readwrite_storage_textures";
  case LanguageFeature::TextureFormatsTier1:
    return "texture_formats_tier1";
  }
  return "?";
}

std::optional<std::string_view> wgslExtension(DeviceFeature f) {
  switch (f) {
  case DeviceFeature::Subgroups:
    return "subgroups";
  case DeviceFeature::ClipDistances:
    return "clip_distances";
  case DeviceFeature::PrimitiveIndex:
    return "primitive_index";
  case DeviceFeature::SubgroupSizeControl:
    return "subgroup_size_control";
  case DeviceFeature::DualSourceBlending:
    return "dual_source_blending";
  case DeviceFeature::ShaderF16:
    return "f16";
  case DeviceFeature::Bgra8UnormStorage:
  case DeviceFeature::Float32Filterable:
  case DeviceFeature::Rg11b10UfloatRenderable:
  case DeviceFeature::TextureFormatsTier1:
  case DeviceFeature::TextureFormatsTier2:
    return std::nullopt;
  }
  return std::nullopt;
}

namespace {

/** What a storage texture access needs: nothing, one feature, or it does not exist. */
enum class Need { Core, Bgra, Tier1, Tier2, Never };

struct FormatRow {
  Need write;
  Need read;
  Need readWrite;
};

/**
 * One row per TexelFormat, in enum order. Measured on Chrome 155 by creating
 * a storage texture bind group layout entry with and without each feature.
 */
constexpr std::array<FormatRow, 40> kFormats = {{
    {Need::Core, Need::Core, Need::Tier2}, // rgba8unorm
    {Need::Core, Need::Core, Need::Never}, // rgba8snorm
    {Need::Core, Need::Core, Need::Tier2}, // rgba8uint
    {Need::Core, Need::Core, Need::Tier2}, // rgba8sint
    {Need::Bgra, Need::Never, Need::Never}, // bgra8unorm: write only, even with the feature
    {Need::Core, Need::Core, Need::Tier2}, // rgba16float
    {Need::Core, Need::Core, Need::Tier2}, // rgba16uint
    {Need::Core, Need::Core, Need::Tier2}, // rgba16sint
    {Need::Core, Need::Core, Need::Tier2}, // rgba32float
    {Need::Core, Need::Core, Need::Tier2}, // rgba32uint
    {Need::Core, Need::Core, Need::Tier2}, // rgba32sint
    {Need::Core, Need::Core, Need::Core}, // r32float
    {Need::Core, Need::Core, Need::Core}, // r32uint
    {Need::Core, Need::Core, Need::Core}, // r32sint
    {Need::Core, Need::Core, Need::Never}, // rg32float
    {Need::Core, Need::Core, Need::Never}, // rg32uint
    {Need::Core, Need::Core, Need::Never}, // rg32sint
    {Need::Tier1, Need::Tier1, Need::Tier2}, // r8unorm
    {Need::Tier1, Need::Tier1, Need::Never}, // r8snorm
    {Need::Tier1, Need::Tier1, Need::Tier2}, // r8uint
    {Need::Tier1, Need::Tier1, Need::Tier2}, // r8sint
    {Need::Tier1, Need::Tier1, Need::Never}, // r16unorm
    {Need::Tier1, Need::Tier1, Need::Never}, // r16snorm
    {Need::Tier1, Need::Tier1, Need::Tier2}, // r16uint
    {Need::Tier1, Need::Tier1, Need::Tier2}, // r16sint
    {Need::Tier1, Need::Tier1, Need::Tier2}, // r16float
    {Need::Tier1, Need::Tier1, Need::Never}, // rg8unorm
    {Need::Tier1, Need::Tier1, Need::Never}, // rg8snorm
    {Need::Tier1, Need::Tier1, Need::Never}, // rg8uint
    {Need::Tier1, Need::Tier1, Need::Never}, // rg8sint
    {Need::Tier1, Need::Tier1, Need::Never}, // rg16unorm
    {Need::Tier1, Need::Tier1, Need::Never}, // rg16snorm
    {Need::Tier1, Need::Tier1, Need::Never}, // rg16uint
    {Need::Tier1, Need::Tier1, Need::Never}, // rg16sint
    {Need::Tier1, Need::Tier1, Need::Never}, // rg16float
    {Need::Tier1, Need::Tier1, Need::Never}, // rgb10a2uint
    {Need::Tier1, Need::Tier1, Need::Never}, // rgb10a2unorm
    {Need::Tier1, Need::Tier1, Need::Never}, // rg11b10ufloat
    {Need::Tier1, Need::Tier1, Need::Never}, // rgba16unorm
    {Need::Tier1, Need::Tier1, Need::Never}, // rgba16snorm
}};

Need needOf(const FormatRow& row, StorageAccess access) {
  switch (access) {
  case StorageAccess::Write:
    return row.write;
  case StorageAccess::Read:
    return row.read;
  case StorageAccess::ReadWrite:
    return row.readWrite;
  }
  return Need::Never;
}

} // namespace

std::optional<std::vector<DeviceFeature>> storageFormatFeatures(TexelFormat format, StorageAccess access) {
  // a value cast from an integer may be outside the enum
  const auto index = static_cast<std::size_t>(format);
  if (index >= std::size(kFormats))
    throw ShaderError(ErrorCode::Internal, std::format("Unknown texel format {}", index));
  switch (needOf(kFormats[index], access)) {
  case Need::Core:
    return std::vector<DeviceFeature>{};
  case Need::Bgra:
    return std::vector{DeviceFeature::Bgra8UnormStorage};
  case Need::Tier1:
    return std::vector{DeviceFeature::Rg11b10UfloatRenderable, DeviceFeature::TextureFormatsTier1};
  case Need::Tier2:
    return std::vector{DeviceFeature::Rg11b10UfloatRenderable, DeviceFeature::TextureFormatsTier1,
                       DeviceFeature::TextureFormatsTier2};
  case Need::Never:
    break;
  }
  return std::nullopt;
}

BackendCapabilities webgpuCapabilities() {
  BackendCapabilities c{"WebGPU", {}};
  c.supported.set();
  return c;
}

BackendCapabilities webgl2Capabilities(bool firstVertexConvention, bool blendFuncExtended) {
  BackendCapabilities c{"WebGL2", {}};
  c.supported.set(static_cast<std::size_t>(Feature::FlatFirstVertex), firstVertexConvention);
  c.supported.set(static_cast<std::size_t>(Feature::DualSourceBlending), blendFuncExtended);
  return c;
}

std::vector<FeatureUse> unsupported(const std::vector<FeatureUse>& uses, const BackendCapabilities& backend) {
  std::vector<FeatureUse> missing;
  for (const FeatureUse& u : uses)
    if (!backend.supports(u.feature)) missing.push_back(u);
  return missing;
}

} // namespace moga
