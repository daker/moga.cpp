#include "Context.hpp"

#include <atomic>
#include <format>

namespace moga {

namespace {

thread_local bool debugEnabled = false;
std::atomic<int> nextLocalId{1};

std::string_view stageText(StageName s) {
  switch (s) {
  case StageName::Vertex:
    return "vertex";
  case StageName::Fragment:
    return "fragment";
  case StageName::Compute:
    return "compute";
  }
  return "?";
}

std::optional<DebugInfo> captureDebug(const Loc& loc) {
  if (!debugEnabled) return std::nullopt;
  return DebugInfo{loc.file_name(), loc.line(), loc.column()};
}

} // namespace

namespace detail {

SourceLocationScope::SourceLocationScope(bool enabled) : previous_(debugEnabled) { debugEnabled = enabled; }
SourceLocationScope::~SourceLocationScope() { debugEnabled = previous_; }

std::vector<RecordedError>*& recordedErrors() {
  thread_local std::vector<RecordedError>* errors = nullptr;
  return errors;
}

std::size_t recordedErrorCount() { return recordedErrors() != nullptr ? recordedErrors()->size() : 0; }

void fail(const ShaderError& error) {
  if (!recovering()) throw error;
  recordedErrors()->push_back({error, currentStage()});
}

std::vector<Block*>& blockStack() {
  thread_local std::vector<Block*> stack;
  return stack;
}

std::vector<Enclosing>& enclosing() {
  thread_local std::vector<Enclosing> stack;
  return stack;
}

std::vector<TypePtr>& functionReturnTypes() {
  thread_local std::vector<TypePtr> types;
  return types;
}

std::optional<StageName>& currentStage() {
  thread_local std::optional<StageName> stage;
  return stage;
}

Block& orphans() {
  thread_local Block block;
  return block;
}

void discardOrphans() { orphans().clear(); }

Stmt& pushStatement(StmtNode node, const Loc& loc) {
  // while recovering, a statement outside any body is kept aside, so the code that adds it goes on
  if (blockStack().empty())
    fail(ShaderError(ErrorCode::OutsideStage, "Shader statement used outside a stage or function body", loc));
  Block& block = blockStack().empty() ? orphans() : *blockStack().back();
  auto stmt = std::make_shared<Stmt>(Stmt{std::move(node), captureDebug(loc)});
  block.push_back(stmt);
  // the next statement's errors name its own operands
  detail::clearUserLocation();
  return *stmt;
}

void requireValueType(const TypePtr& type, std::string_view what) {
  if (type == voidType()) throw ShaderError(ErrorCode::InvalidType, std::format("{} cannot be void", what));
  if (containsAtomic(type))
    throw ShaderError(ErrorCode::InvalidType, std::format("{} cannot be or hold an atomic ({}); atomics live in storage and "
                                  "workgroup memory",
                                  what, typeName(type)));
}

void requireStage(StageName stage, std::string_view what) {
  if (currentStage() != stage)
    throw ShaderError(ErrorCode::WrongStage, std::format("{} is only valid in the {} stage", what, stageText(stage)));
}

bool buildingDetachedFunction() { return !currentStage() && !functionReturnTypes().empty(); }

void checkType(const Val& v, const TypePtr& type, std::string_view what) {
  if (!typesEqual(v.type(), type))
    throw ShaderError(ErrorCode::TypeMismatch, 
        std::format("{}: expected {}, got {}", what, typeName(type), typeName(v.type())));
}

LocalPtr newLocal(std::string name, TypePtr type, bool mutableVar) {
  return std::make_shared<const LocalVar>(
      LocalVar{nextLocalId++, std::move(name), std::move(type), mutableVar});
}

Val localVal(const LocalPtr& v) { return Val(makeExpr(ex::Local{v}, v->type)); }

Val literal(std::variant<double, bool> value, const TypePtr& type) {
  return Val(makeExpr(ex::Lit{value}, type));
}

Val integerScalar(const ValueLike& v, std::string_view what) {
  const Val x = toVal(v, i32());
  if (x.type() != i32() && x.type() != u32())
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{} must be i32 or u32, got {}", what, typeName(x.type())));
  return x;
}

bool isScalarOrVector(const TypePtr& t) {
  return t->kind == TypeKind::Scalar || t->kind == TypeKind::Vector;
}

Val coerceFirst(const ValueLike& a, const ValueLike& b) {
  if (const Val* v = a.val()) return *v;
  if (const Val* v = b.val()) return toVal(a, v->type());
  return toVal(a, f32());
}

} // namespace detail

} // namespace moga
