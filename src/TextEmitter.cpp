#include "TextEmitter.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <set>

namespace moga::detail {

namespace {
// Names that no identifier may take: the GLSL ES 3.00 keywords and reserved
// words (section 3.6) and the WGSL keywords, type names and reserved words. A
// clash gets a trailing underscore.
const std::set<std::string, std::less<>> kReserved = {
    "NULL", "Self", "abstract", "active", "alias", "alignas", "alignof", "array", "as", "asm",
    "asm_fragment", "async", "atomic", "atomic_uint", "attribute", "auto", "await", "become",
    "binding_array", "bool", "break", "buffer", "bvec2", "bvec3", "bvec4", "case", "cast",
    "catch", "centroid", "class", "co_await", "co_return", "co_yield", "coherent", "column_major",
    "common", "compile", "compile_fragment", "concept", "const", "const_assert", "const_cast",
    "consteval", "constexpr", "constinit", "continue", "continuing", "crate", "debugger",
    "decltype", "default", "delete", "demote", "demote_to_helper", "diagnostic", "discard", "do",
    "double", "dvec2", "dvec3", "dvec4", "dynamic_cast", "else", "enable", "enum", "explicit",
    "export", "extends", "extern", "external", "f16", "f32", "fallthrough", "false", "filter",
    "final", "finally", "fixed", "flat", "float", "fn", "for", "friend", "from", "function",
    "fvec2", "fvec3", "fvec4", "fxgroup", "get", "goto", "groupshared", "half", "highp", "hvec2",
    "hvec3", "hvec4", "i32", "if", "iimage1D", "iimage1DArray", "iimage2D", "iimage2DArray",
    "iimage3D", "iimageBuffer", "iimageCube", "image1D", "image1DArray", "image2D",
    "image2DArray", "image3D", "imageBuffer", "imageCube", "impl", "implements", "import", "in",
    "inline", "inout", "input", "instanceof", "int", "interface", "invariant", "isampler1D",
    "isampler1DArray", "isampler2D", "isampler2DArray", "isampler2DMS", "isampler2DMSArray",
    "isampler2DRect", "isampler3D", "isamplerBuffer", "isamplerCube", "ivec2", "ivec3", "ivec4",
    "layout", "let", "long", "loop", "lowp", "macro", "macro_rules", "mat", "mat2", "mat2x2",
    "mat2x2f", "mat2x2h", "mat2x3", "mat2x3f", "mat2x3h", "mat2x4", "mat2x4f", "mat2x4h", "mat3",
    "mat3x2", "mat3x2f", "mat3x2h", "mat3x3", "mat3x3f", "mat3x3h", "mat3x4", "mat3x4f", "mat3x4h",
    "mat4", "mat4x2", "mat4x2f", "mat4x2h", "mat4x3", "mat4x3f", "mat4x3h", "mat4x4", "mat4x4f",
    "mat4x4h", "match", "mediump", "meta", "module", "move", "mut", "mutable", "namespace", "new",
    "nil", "noexcept", "noinline", "nointerpolation", "noperspective", "null", "nullptr", "of",
    "operator", "out", "output", "override", "package", "packed", "packoffset", "partition",
    "pass", "patch", "pixelfragment", "precise", "precision", "premerge", "priv", "private",
    "protected", "ptr", "pub", "public", "read", "read_write", "readonly", "ref", "regardless",
    "register", "reinterpret_cast", "require", "requires", "resource", "restrict", "return",
    "sample", "sampler", "sampler1D", "sampler1DArray", "sampler1DArrayShadow", "sampler1DShadow",
    "sampler2D", "sampler2DArray", "sampler2DArrayShadow", "sampler2DMS", "sampler2DMSArray",
    "sampler2DRect", "sampler2DRectShadow", "sampler2DShadow", "sampler3D", "sampler3DRect",
    "samplerBuffer", "samplerCube", "samplerCubeShadow", "sampler_comparison", "self", "set",
    "shared", "short", "sizeof", "smooth", "snorm", "static", "static_assert", "static_cast",
    "std", "storage", "struct", "subroutine", "super", "superp", "switch", "target", "template",
    "texture", "texture_1d", "texture_2d", "texture_2d_array", "texture_3d", "texture_cube",
    "texture_cube_array", "texture_depth_2d", "texture_depth_2d_array", "texture_depth_cube",
    "texture_depth_cube_array", "texture_depth_multisampled_2d", "texture_external",
    "texture_multisampled_2d", "texture_storage_1d", "texture_storage_2d",
    "texture_storage_2d_array", "texture_storage_3d", "this", "thread_local", "throw", "trait",
    "true", "try", "type", "typedef", "typeid", "typename", "typeof", "u32", "uimage1D",
    "uimage1DArray", "uimage2D", "uimage2DArray", "uimage3D", "uimageBuffer", "uimageCube",
    "uint", "uniform", "union", "unless", "unorm", "unsafe", "unsigned", "unsized", "usampler1D",
    "usampler1DArray", "usampler2D", "usampler2DArray", "usampler2DMS", "usampler2DMSArray",
    "usampler2DRect", "usampler3D", "usamplerBuffer", "usamplerCube", "use", "using", "uvec2",
    "uvec3", "uvec4", "var", "varying", "vec", "vec2", "vec2f", "vec2h", "vec2i", "vec2u", "vec3",
    "vec3f", "vec3h", "vec3i", "vec3u", "vec4", "vec4f", "vec4h", "vec4i", "vec4u", "virtual",
    "void", "volatile", "wgsl", "where", "while", "with", "workgroup", "write", "writeonly",
    "yield"};

// Builtin functions of both languages and GLSL main: a module scope
// declaration with one of these names would hide the builtin. Only the bare
// names (constants) can clash; prefixed names cannot.
const std::set<std::string, std::less<>> kBuiltinFunctions = {
    "abs", "acos", "acosh", "all", "any", "arrayLength", "asin", "asinh", "atan", "atan2",
    "atanh", "atomicAdd", "atomicAnd", "atomicCompareExchangeWeak", "atomicExchange",
    "atomicLoad", "atomicMax", "atomicMin", "atomicOr", "atomicStore", "atomicSub", "atomicXor",
    "bitcast", "ceil", "clamp", "cos", "cosh", "countLeadingZeros", "countOneBits",
    "countTrailingZeros", "cross", "dFdx", "dFdy", "degrees", "determinant", "distance", "dot",
    "dpdx", "dpdxCoarse", "dpdxFine", "dpdy", "dpdyCoarse", "dpdyFine", "equal", "exp", "exp2",
    "extractBits", "faceForward", "faceforward", "firstLeadingBit", "firstTrailingBit",
    "floatBitsToInt", "floatBitsToUint", "floor", "fma", "fract", "frexp", "fwidth",
    "fwidthCoarse", "fwidthFine", "greaterThan", "greaterThanEqual", "insertBits",
    "intBitsToFloat", "inverse", "inverseSqrt", "inversesqrt", "isinf", "isnan", "ldexp",
    "length", "lessThan", "lessThanEqual", "log", "log2", "main", "matrixCompMult", "max", "min",
    "mix", "mod", "modf", "normalize", "not", "notEqual", "outerProduct", "pack2x16float",
    "pack2x16snorm", "pack2x16unorm", "pack4x8snorm", "pack4x8unorm", "packHalf2x16",
    "packSnorm2x16", "packUnorm2x16", "pow", "quantizeToF16", "radians", "reflect", "refract",
    "reverseBits", "round", "roundEven", "saturate", "select", "sign", "sin", "sinh", "smoothstep", "sqrt",
    "step", "storageBarrier", "subgroupAdd", "subgroupAll", "subgroupAnd", "subgroupAny",
    "subgroupBallot", "subgroupBroadcast", "subgroupBroadcastFirst", "subgroupElect",
    "subgroupMax", "subgroupMin", "subgroupMul", "subgroupOr", "subgroupShuffle", "subgroupXor",
    "tan", "tanh", "texelFetch", "texture", "texelFetchOffset", "textureBarrier", "textureDimensions",
    "textureGather", "textureGatherCompare", "textureGrad", "textureGradOffset", "textureLoad", "textureLod",
    "textureLodOffset", "textureNumLayers", "textureNumLevels", "textureNumSamples",
    "textureOffset", "textureProj", "textureProjGrad", "textureProjGradOffset", "textureProjLod",
    "textureProjLodOffset", "textureProjOffset", "textureSample", "textureSampleBaseClampToEdge",
    "textureSampleBias", "textureSampleCompare", "textureSampleCompareLevel", "textureSampleGrad",
    "textureSampleLevel", "textureSize", "textureStore", "transpose", "trunc", "uintBitsToFloat",
    "unpack2x16float", "unpack2x16snorm", "unpack2x16unorm", "unpack4x8snorm", "unpack4x8unorm",
    "unpackHalf2x16", "unpackSnorm2x16", "unpackUnorm2x16", "workgroupBarrier",
    "workgroupUniformLoad"};
} // namespace

std::string sanitize(std::string_view name) {
  std::string s;
  for (char c : name) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    const char out = ok ? c : '_';
    if (out == '_' && !s.empty() && s.back() == '_') continue;
    s += out;
  }
  // GLSL reserves the gl_ prefix and WebGL the webgl_ prefix; GL_ names are predefined macros
  // (GL_ES, GL_FRAGMENT_PRECISION_HIGH, one per enabled extension) that would replace the name
  if (s.empty() || !std::isalpha(static_cast<unsigned char>(s.front())) || s.starts_with("gl_") ||
      s.starts_with("webgl_") || s.starts_with("GL_"))
    s = "x" + s;
  return kReserved.contains(s) ? s + "_" : s;
}

std::string constantName(std::string_view name) {
  const std::string s = sanitize(name);
  // locals print as name_<id> (localName), so a constant that ends the same way could be hidden by one
  const std::size_t digits = s.size() - s.find_last_not_of("0123456789") - 1;
  const bool localLike = digits > 0 && digits < s.size() && s[s.size() - digits - 1] == '_';
  return kBuiltinFunctions.contains(s) || localLike ? s + "_" : s;
}

std::string typeTag(std::string_view glslOrWgslType) {
  std::string tag;
  for (const char c : glslOrWgslType) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) != 0;
    if (ok) tag += c;
    else if (!tag.empty() && tag.back() != '_') tag += '_';
  }
  while (!tag.empty() && tag.back() == '_') tag.pop_back();
  return tag;
}

std::string localName(const LocalVar& v) {
  // GLSL reserves names with two underscores in a row, and sanitize() can end with one
  std::string base = sanitize(v.name);
  // generated locals start with moga_ (moga_shared_3), so a user local never does
  if (base.starts_with("moga")) base = "l_" + base;
  std::string name = std::format("{}{}{}", base, base.ends_with('_') ? "" : "_", v.id);
  // module scope names start with a prefix and may end in digits (f_foo_1 for function foo_1,
  // o_0 for output 0), so a local that starts the same way could hide one
  static constexpr std::array<std::string_view, 13> kModulePrefixes{
      "S_", "UB_", "a_", "f_", "o_", "s_", "sb_", "st_", "t_", "tb_", "ub_", "v_", "wg_"};
  if (std::ranges::any_of(kModulePrefixes, [&](std::string_view p) { return name.starts_with(p); }))
    name = "l_" + name;
  return name;
}
std::string functionName(const FunctionDecl& f) { return "f_" + sanitize(f.name); }
std::string structName(std::string_view name) { return "S_" + sanitize(name); }

std::string formatFloat(double v) {
  if (!std::isfinite(v)) throw ShaderError(ErrorCode::InvalidConstant, std::format("Non-finite shader literal {}", v));
  // the shortest decimal of a double is near it, not equal to it: for a value an f32 (or f16)
  // holds exactly, a nearby decimal would let WGSL pick a neighbor, or step past the largest f32.
  // Every binary fraction has an exact decimal; an f32 needs at most 112 significant digits
  const bool f32Exact = static_cast<double>(static_cast<float>(v)) == v;
  constexpr int kExactDigits = 120;
  std::string s = f32Exact ? std::format("{:.{}g}", v, kExactDigits) : std::format("{}", v);
  if (s.find_first_of(".e") != std::string::npos) return s;
  return s + ".0";
}

std::string join(const std::vector<std::string>& parts, std::string_view sep) {
  std::string out;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) out += sep;
    out += parts[i];
  }
  return out;
}

void TextEmitter::line(std::string_view text, const std::optional<DebugInfo>& debug) {
  lines_.push_back(text.empty() ? std::string() : std::string(indent_ * 2, ' ') + std::string(text));
  if (debug) mapping.push_back({static_cast<int>(lines_.size()), *debug});
}

std::string TextEmitter::str() const {
  std::string out;
  for (const std::string& l : lines_) {
    out += l;
    out += '\n';
  }
  return out;
}

} // namespace moga::detail
