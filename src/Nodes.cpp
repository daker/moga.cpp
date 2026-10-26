#include "moga/Nodes.hpp"
#include "BuiltinTable.hpp"
#include "Walk.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <format>

namespace moga {

namespace {
std::atomic<int> nextNodeId{1};

constexpr std::array<std::string_view, 14> kInputNames = {
    "vertexIndex", "instanceIndex", "localInstanceIndex", "drawIndex", "fragCoord",
    "frontFacing", "globalId", "localId", "workgroupId", "numWorkgroups", "localIndex",
    "subgroupSize", "subgroupInvocationId", "primitiveIndex"};

constexpr std::array<std::string_view, 9> kAtomicNames = {
    "atomicLoad", "atomicAdd", "atomicSub", "atomicMax", "atomicMin",
    "atomicAnd",  "atomicOr",  "atomicXor", "atomicExchange"};

constexpr std::array<std::string_view, 40> kFormatNames = {
    "rgba8unorm",   "rgba8snorm",    "rgba8uint",   "rgba8sint",   "bgra8unorm", "rgba16float",
    "rgba16uint",   "rgba16sint",    "rgba32float", "rgba32uint",  "rgba32sint", "r32float",
    "r32uint",      "r32sint",       "rg32float",   "rg32uint",    "rg32sint",   "r8unorm",
    "r8snorm",      "r8uint",        "r8sint",      "r16unorm",    "r16snorm",   "r16uint",
    "r16sint",      "r16float",      "rg8unorm",    "rg8snorm",    "rg8uint",    "rg8sint",
    "rg16unorm",    "rg16snorm",     "rg16uint",    "rg16sint",    "rg16float",  "rgb10a2uint",
    "rgb10a2unorm", "rg11b10ufloat", "rgba16unorm", "rgba16snorm"};

constexpr std::array<std::string_view, 18> kBinaryText = {
    "+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "&&", "||", "&", "|", "^",
    "<<", ">>"};
} // namespace

std::string_view builtinName(BuiltinFunction fn) { return detail::builtinSpec(fn).name; }
std::string_view builtinInputName(BuiltinInput b) { return kInputNames[static_cast<int>(b)]; }
std::string_view atomicOpName(AtomicOp op) { return kAtomicNames[static_cast<int>(op)]; }
std::string outputName(int index, std::optional<int> blendSrc) {
  return blendSrc == 1 ? std::format("o_{}_src1", index) : std::format("o_{}", index);
}

std::string_view texelFormatName(TexelFormat f) {
  // a value cast from an integer may be outside the enum
  const auto index = static_cast<std::size_t>(f);
  if (index >= kFormatNames.size()) throw ShaderError(ErrorCode::Internal, std::format("Unknown texel format {}", index));
  return kFormatNames[index];
}

ScalarKind texelFormatScalar(TexelFormat f) {
  const std::string_view name = texelFormatName(f);
  if (name.ends_with("uint")) return ScalarKind::U32;
  if (name.ends_with("sint")) return ScalarKind::I32;
  return ScalarKind::F32;
}

bool isComputeBuiltin(BuiltinInput b) {
  return b == BuiltinInput::GlobalId || b == BuiltinInput::LocalId ||
         b == BuiltinInput::WorkgroupId || b == BuiltinInput::NumWorkgroups ||
         b == BuiltinInput::LocalIndex;
}

bool isSubgroupBuiltin(BuiltinInput b) {
  return b == BuiltinInput::SubgroupSize || b == BuiltinInput::SubgroupInvocationId;
}

bool isSubgroupFunction(BuiltinFunction fn) { return detail::builtinSpec(fn).subgroup; }

std::string_view opText(BinaryOp op) { return kBinaryText[static_cast<int>(op)]; }

std::string_view opText(UnaryOp op) {
  switch (op) {
  case UnaryOp::Neg:
    return "-";
  case UnaryOp::Not:
    return "!";
  case UnaryOp::BitNot:
    return "~";
  }
  return "?";
}

bool hasSideEffects(const Expr& e) { return e.sideEffects; }

namespace {

/** Nodes that combine their operands without reading memory, inputs or locals. */
bool combinesOperands(const ExprNode& n) {
  return std::holds_alternative<ex::Binary>(n) || std::holds_alternative<ex::Unary>(n) ||
         std::holds_alternative<ex::Construct>(n) || std::holds_alternative<ex::Call>(n) ||
         std::holds_alternative<ex::Swizzle>(n) || std::holds_alternative<ex::Member>(n) ||
         std::holds_alternative<ex::Index>(n) || std::holds_alternative<ex::Select>(n);
}

bool readsMutable(const ExprNode& n) {
  if (const auto* s = std::get_if<ex::StorageElement>(&n)) return s->buffer->access != StorageAccess::Read;
  if (const auto* l = std::get_if<ex::StorageLoad>(&n)) return l->tex->access != StorageAccess::Read;
  return std::holds_alternative<ex::Workgroup>(n) || std::holds_alternative<ex::Atomic>(n);
}

/** Derivatives and subgroup operations must run in uniform control flow. */
bool needsUniformFlowCall(const ExprNode& n) {
  const auto* c = std::get_if<ex::Call>(&n);
  if (c == nullptr) return false;
  const detail::BuiltinSpec& spec = detail::builtinSpec(c->fn);
  return spec.stages == detail::Stages::Fragment || spec.subgroup;
}

} // namespace

ExprPtr makeExpr(ExprNode node, TypePtr type) {
  Expr e{nextNodeId++, std::move(type), std::move(node)};
  // a user call can write storage and an atomic changes memory; the other facts
  // follow from the operands' facts, one look per operand however often a node is reused
  const bool isConstant = std::holds_alternative<ex::Constant>(e.node);
  e.sideEffects = std::holds_alternative<ex::UCall>(e.node) || std::holds_alternative<ex::Atomic>(e.node);
  e.needsUniformity = std::holds_alternative<ex::UCall>(e.node) || needsUniformFlowCall(e.node);
  e.constantLeaves = std::holds_alternative<ex::Lit>(e.node) || isConstant || combinesOperands(e.node);
  e.readsMutableMemory = readsMutable(e.node);
  // a named constant is a leaf, but one made from a poisoned value is poisoned too
  if (const auto* c = std::get_if<ex::Constant>(&e.node)) e.poisoned = c->c->value->poisoned;
  if (!isConstant) {
    detail::forEachOperand(e, [&](const Expr& operand) {
      e.sideEffects = e.sideEffects || operand.sideEffects;
      e.needsUniformity = e.needsUniformity || operand.needsUniformity;
      e.readsMutableMemory = e.readsMutableMemory || operand.readsMutableMemory;
      e.constantLeaves = e.constantLeaves && operand.constantLeaves;
      e.treeSize = std::min(kTreeSizeLimit, e.treeSize + operand.treeSize);
      e.poisoned = e.poisoned || operand.poisoned;
    });
  }
  return std::make_shared<const Expr>(std::move(e));
}

ExprPtr makePoison(TypePtr type) {
  Expr e{nextNodeId++, std::move(type), ex::Lit{0.0}};
  e.poisoned = true;
  // not a constant: no check may evaluate it
  e.constantLeaves = false;
  return std::make_shared<const Expr>(std::move(e));
}

} // namespace moga
