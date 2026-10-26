#include "moga/Types.hpp"

#include "TextEmitter.hpp"
#include "dsl/Context.hpp"

#include <array>
#include <algorithm>
#include <format>
#include <set>
#include <utility>

namespace moga {

namespace {

/** The last place where user code handed a value to the DSL, on this thread. */
thread_local std::optional<std::source_location> userLocation;

/** Whether `path` contains `segment`, where '/' in the segment also matches '\\'. */
bool containsSegment(std::string_view path, std::string_view segment) {
  for (std::size_t i = 0; i + segment.size() <= path.size(); ++i) {
    std::size_t k = 0;
    for (; k < segment.size(); ++k) {
      const char c = path[i + k];
      if (segment[k] == '/' ? (c != '/' && c != '\\') : c != segment[k]) break;
    }
    if (k == segment.size()) return true;
  }
  return false;
}

/**
 * Whether a location is in moga's own sources or headers. A translation unit
 * passes the same file name pointer each time, so the last answers are kept:
 * the check runs for every operand and must cost next to nothing.
 */
bool insideMoga(const char* file) {
  struct Entry {
    const char* file = nullptr;
    bool inside = false;
  };
  thread_local std::array<Entry, 8> recent{};
  thread_local std::size_t next = 0;
  for (const Entry& e : recent)
    if (e.file == file) return e.inside;
  const std::string_view path(file);
  const bool inside = containsSegment(path, "/include/moga/") || containsSegment(path, "/moga/src/");
  recent[next] = {file, inside};
  next = (next + 1) % recent.size();
  return inside;
}

std::string located(const std::string& message, const std::optional<DebugInfo>& where) {
  if (!where) return message;
  return std::format("{}:{}:{}: {}", where->file, where->line, where->column, message);
}

std::optional<DebugInfo> toDebugInfo(const std::optional<std::source_location>& at) {
  if (!at) return std::nullopt;
  return DebugInfo{at->file_name(), at->line(), at->column()};
}

constexpr std::array kErrorCodeNames{
    "TypeMismatch",     "InvalidOperands", "ArgumentCount", "InvalidType",   "UnknownName",
    "DuplicateName",    "ReservedName",    "WrongStage",    "NotAssignable", "OutOfRange",
    "ConstantRequired", "InvalidConstant", "ControlFlow",   "StageInterface", "Layout",
    "TextureUsage",     "InvalidDeclaration", "Recursion",  "Uniformity",    "Scope",
    "MissingFeature",   "OutsideStage",    "UniformWriter", "Driver",
    "Internal"};
static_assert(kErrorCodeNames.size() == static_cast<std::size_t>(ErrorCode::Internal) + 1,
              "one name per error code");

} // namespace

std::string_view errorCodeName(ErrorCode code) { return kErrorCodeNames.at(static_cast<std::size_t>(code)); }

ShaderError::ShaderError(ErrorCode code, const std::string& message)
    : ShaderError(code, message, toDebugInfo(userLocation)) {}

ShaderError::ShaderError(ErrorCode code, const std::string& message, const std::source_location& at)
    : ShaderError(code, message, toDebugInfo(insideMoga(at.file_name()) ? userLocation : std::optional(at))) {}

ShaderError::ShaderError(ErrorCode code, const std::string& message, std::optional<DebugInfo> where)
    : std::runtime_error(located(message, where)), code_(code), message_(message), where_(std::move(where)) {}

namespace detail {

void noteUserLocation(const std::source_location& loc) {
  if (!insideMoga(loc.file_name())) userLocation = loc;
}

void clearUserLocation() { userLocation.reset(); }

} // namespace detail


namespace {

TypePtr makeScalar(ScalarKind s) {
  auto t = std::make_shared<TypeInfo>();
  t->kind = TypeKind::Scalar;
  t->scalar = s;
  return t;
}

TypePtr makeVector(ScalarKind s, int size) {
  auto t = std::make_shared<TypeInfo>();
  t->kind = TypeKind::Vector;
  t->scalar = s;
  t->size = size;
  return t;
}

TypePtr makeAtomic(ScalarKind s) {
  auto t = std::make_shared<TypeInfo>();
  t->kind = TypeKind::Atomic;
  t->scalar = s;
  return t;
}

TypePtr makeMatrix(int cols, int rows, ScalarKind s) {
  auto t = std::make_shared<TypeInfo>();
  t->kind = TypeKind::Matrix;
  t->scalar = s;
  t->cols = cols;
  t->rows = rows;
  return t;
}

struct Interned {
  std::array<TypePtr, 5> scalars{makeScalar(ScalarKind::F32), makeScalar(ScalarKind::I32),
                                 makeScalar(ScalarKind::U32), makeScalar(ScalarKind::Bool),
                                 makeScalar(ScalarKind::F16)};
  std::array<std::array<TypePtr, 3>, 5> vectors{};
  std::array<TypePtr, 3> matrices{makeMatrix(2, 2, ScalarKind::F32), makeMatrix(3, 3, ScalarKind::F32),
                                  makeMatrix(4, 4, ScalarKind::F32)};
  std::array<TypePtr, 3> halfMatrices{makeMatrix(2, 2, ScalarKind::F16), makeMatrix(3, 3, ScalarKind::F16),
                                      makeMatrix(4, 4, ScalarKind::F16)};
  TypePtr voidT = std::make_shared<TypeInfo>();
  TypePtr atomicU = makeAtomic(ScalarKind::U32);
  TypePtr atomicI = makeAtomic(ScalarKind::I32);

  Interned() {
    for (int s = 0; s < 5; ++s)
      for (int n = 2; n <= 4; ++n) vectors[s][n - 2] = makeVector(static_cast<ScalarKind>(s), n);
  }
};

const Interned& interned() {
  static const Interned instance;
  return instance;
}

int idx(ScalarKind k) { return static_cast<int>(k); }

} // namespace

const TypePtr& scalarType(ScalarKind kind) { return interned().scalars[idx(kind)]; }

const TypePtr& vectorType(ScalarKind kind, int size) {
  if (size == 1) return scalarType(kind);
  if (size < 2 || size > 4) {
    detail::fail(ShaderError(ErrorCode::InvalidType, std::format("Unsupported vector size {}", size)));
    size = std::clamp(size, 2, 4);
  }
  return interned().vectors[idx(kind)][size - 2];
}

const TypePtr& squareMatrix(int n, ScalarKind kind) {
  // while recovering, the nearest valid matrix stands in
  if (n < 2 || n > 4) {
    detail::fail(ShaderError(ErrorCode::InvalidType, std::format("Unsupported matrix size {}", n)));
    n = std::clamp(n, 2, 4);
  }
  if (!isFloatKind(kind)) {
    detail::fail(ShaderError(ErrorCode::InvalidType, "Matrices hold f32 or f16"));
    kind = ScalarKind::F32;
  }
  return kind == ScalarKind::F16 ? interned().halfMatrices[n - 2] : interned().matrices[n - 2];
}

const TypePtr& f32() { return scalarType(ScalarKind::F32); }
const TypePtr& i32() { return scalarType(ScalarKind::I32); }
const TypePtr& u32() { return scalarType(ScalarKind::U32); }
const TypePtr& boolType() { return scalarType(ScalarKind::Bool); }
const TypePtr& f16() { return scalarType(ScalarKind::F16); }
const TypePtr& vec2f() { return vectorType(ScalarKind::F32, 2); }
const TypePtr& vec3f() { return vectorType(ScalarKind::F32, 3); }
const TypePtr& vec4f() { return vectorType(ScalarKind::F32, 4); }
const TypePtr& vec2i() { return vectorType(ScalarKind::I32, 2); }
const TypePtr& vec3i() { return vectorType(ScalarKind::I32, 3); }
const TypePtr& vec4i() { return vectorType(ScalarKind::I32, 4); }
const TypePtr& vec2u() { return vectorType(ScalarKind::U32, 2); }
const TypePtr& vec3u() { return vectorType(ScalarKind::U32, 3); }
const TypePtr& vec4u() { return vectorType(ScalarKind::U32, 4); }
const TypePtr& vec2b() { return vectorType(ScalarKind::Bool, 2); }
const TypePtr& vec3b() { return vectorType(ScalarKind::Bool, 3); }
const TypePtr& vec4b() { return vectorType(ScalarKind::Bool, 4); }
const TypePtr& vec2h() { return vectorType(ScalarKind::F16, 2); }
const TypePtr& vec3h() { return vectorType(ScalarKind::F16, 3); }
const TypePtr& vec4h() { return vectorType(ScalarKind::F16, 4); }
const TypePtr& mat2f() { return squareMatrix(2); }
const TypePtr& mat3f() { return squareMatrix(3); }
const TypePtr& mat4f() { return squareMatrix(4); }
const TypePtr& mat2h() { return squareMatrix(2, ScalarKind::F16); }
const TypePtr& mat3h() { return squareMatrix(3, ScalarKind::F16); }
const TypePtr& mat4h() { return squareMatrix(4, ScalarKind::F16); }
const TypePtr& voidType() { return interned().voidT; }
const TypePtr& atomicU32() { return interned().atomicU; }
const TypePtr& atomicI32() { return interned().atomicI; }

TypePtr arrayOf(TypePtr element, int length, std::source_location loc) {
  detail::noteUserLocation(loc);
  // while recovering, the array is made with a length of 1 and f32 elements in place of invalid ones
  if (length <= 0) {
    detail::fail(ShaderError(ErrorCode::InvalidType, std::format("array<{}, {}>: the length must be positive", typeName(element), length)));
    length = 1;
  }
  if (element->kind == TypeKind::Void) {
    detail::fail(ShaderError(ErrorCode::InvalidType, "An array cannot hold void"));
    element = f32();
  }
  auto t = std::make_shared<TypeInfo>();
  t->kind = TypeKind::Array;
  t->element = std::move(element);
  t->length = length;
  return t;
}

TypePtr makeStruct(std::string name, const StructFields& fields, std::source_location loc) {
  detail::noteUserLocation(loc);
  auto t = std::make_shared<TypeInfo>();
  t->kind = TypeKind::Struct;
  t->name = std::move(name);
  // while recovering, a field that fails is left out
  if (fields.empty())
    detail::fail(ShaderError(ErrorCode::InvalidDeclaration, std::format("Struct {} needs at least one field", t->name)));
  std::set<std::string> emitted;
  for (const auto& [n, type] : fields) {
    if (type->kind == TypeKind::Void) {
      detail::fail(ShaderError(ErrorCode::InvalidType, std::format("Struct {}: field {} cannot be void", t->name, n)));
      continue;
    }
    // the backends emit sanitized field names, so those must differ too
    if (!emitted.insert(detail::sanitize(n)).second) {
      detail::fail(ShaderError(ErrorCode::DuplicateName, std::format("Struct {}: field {} repeats another field name", t->name, n)));
      continue;
    }
    t->fields.push_back({n, type});
  }
  return t;
}

bool typesEqual(const TypePtr& a, const TypePtr& b) {
  if (a == b) return true;
  if (a->kind == TypeKind::Array && b->kind == TypeKind::Array)
    return a->length == b->length && typesEqual(a->element, b->element);
  if (a->kind == TypeKind::Struct && b->kind == TypeKind::Struct) {
    // the backends declare structs by name, so equal structs need the same name too
    if (a->name != b->name || a->fields.size() != b->fields.size()) return false;
    for (std::size_t i = 0; i < a->fields.size(); ++i)
      if (a->fields[i].name != b->fields[i].name || !typesEqual(a->fields[i].type, b->fields[i].type))
        return false;
    return true;
  }
  return false;
}

ScalarKind scalarOf(const TypePtr& t) {
  switch (t->kind) {
  case TypeKind::Scalar:
  case TypeKind::Vector:
    return t->scalar;
  case TypeKind::Matrix:
    return t->scalar;
  default:
    throw ShaderError(ErrorCode::Internal, std::format("Type {} has no scalar kind", typeName(t)));
  }
}

int componentCount(const TypePtr& t) {
  switch (t->kind) {
  case TypeKind::Scalar:
    return 1;
  case TypeKind::Vector:
    return t->size;
  case TypeKind::Matrix:
    return t->cols * t->rows;
  default:
    throw ShaderError(ErrorCode::Internal, std::format("Type {} has no component count", typeName(t)));
  }
}

static bool isScalarOrVector(const TypePtr& t) {
  return t->kind == TypeKind::Scalar || t->kind == TypeKind::Vector;
}

bool isNumeric(const TypePtr& t) { return isScalarOrVector(t) && t->scalar != ScalarKind::Bool; }

bool isFloatKind(ScalarKind k) { return k == ScalarKind::F32 || k == ScalarKind::F16; }

bool isFloat(const TypePtr& t) {
  return t->kind == TypeKind::Matrix || (isScalarOrVector(t) && isFloatKind(t->scalar));
}

bool containsF16(const TypePtr& t) {
  if (t->kind == TypeKind::Array) return containsF16(t->element);
  if (t->kind == TypeKind::Struct)
    return std::ranges::any_of(t->fields, [](const StructField& f) { return containsF16(f.type); });
  const bool valued = t->kind == TypeKind::Scalar || t->kind == TypeKind::Vector || t->kind == TypeKind::Matrix;
  return valued && t->scalar == ScalarKind::F16;
}

bool isInteger(const TypePtr& t) {
  return isScalarOrVector(t) && (t->scalar == ScalarKind::I32 || t->scalar == ScalarKind::U32);
}

bool isBool(const TypePtr& t) { return isScalarOrVector(t) && t->scalar == ScalarKind::Bool; }

static std::string_view scalarName(ScalarKind k) {
  switch (k) {
  case ScalarKind::F32:
    return "f32";
  case ScalarKind::I32:
    return "i32";
  case ScalarKind::U32:
    return "u32";
  case ScalarKind::Bool:
    return "bool";
  case ScalarKind::F16:
    return "f16";
  }
  return "?";
}

bool containsAtomic(const TypePtr& t) {
  if (t->kind == TypeKind::Atomic) return true;
  if (t->kind == TypeKind::Array) return containsAtomic(t->element);
  if (t->kind == TypeKind::Struct)
    for (const StructField& f : t->fields)
      if (containsAtomic(f.type)) return true;
  return false;
}

std::string typeName(const TypePtr& t) {
  switch (t->kind) {
  case TypeKind::Scalar:
    return std::string(scalarName(t->scalar));
  case TypeKind::Vector:
    if (t->scalar == ScalarKind::Bool) return std::format("vec{}<bool>", t->size);
    return std::format("vec{}{}", t->size, t->scalar == ScalarKind::F16 ? 'h' : scalarName(t->scalar)[0]);
  case TypeKind::Matrix:
    return std::format("mat{}x{}{}", t->cols, t->rows, t->scalar == ScalarKind::F16 ? 'h' : 'f');
  case TypeKind::Array:
    return std::format("array<{}, {}>", typeName(t->element), t->length);
  case TypeKind::Struct:
    return t->name;
  case TypeKind::Atomic:
    return std::format("atomic<{}>", scalarName(t->scalar));
  case TypeKind::Void:
    return "void";
  }
  return "?";
}

} // namespace moga
