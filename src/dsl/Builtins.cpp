#include "Context.hpp"
#include "moga/dsl/Builtins.hpp"

#include <format>

namespace moga {

using namespace detail;

static Val builtin(BuiltinInput name, const TypePtr& type, StageName stage) {
  requireStage(stage, std::format("Builtin {}", builtinInputName(name)));
  return Val(makeExpr(ex::Builtin{name}, type));
}

Val vertexIndex() {
  return detail::guard([&]() -> Val { return builtin(BuiltinInput::VertexIndex, u32(), StageName::Vertex); });
}
Val instanceIndex() {
  return detail::guard([&]() -> Val { return builtin(BuiltinInput::InstanceIndex, u32(), StageName::Vertex); });
}
Val localInstanceIndex() {
  return detail::guard([&]() -> Val {
    return builtin(BuiltinInput::LocalInstanceIndex, u32(), StageName::Vertex);
  });
}
Val drawIndex() {
  return detail::guard([&]() -> Val { return builtin(BuiltinInput::DrawIndex, u32(), StageName::Vertex); });
}
Val fragCoord() {
  return detail::guard([&]() -> Val { return builtin(BuiltinInput::FragCoord, vec4f(), StageName::Fragment); });
}
Val frontFacing() {
  return detail::guard([&]() -> Val { return builtin(BuiltinInput::FrontFacing, boolType(), StageName::Fragment); });
}
Val primitiveIndex() {
  return detail::guard([&]() -> Val { return builtin(BuiltinInput::PrimitiveIndex, u32(), StageName::Fragment); });
}

void Position(const Val& value, Loc loc) {
  return detail::guard([&] {
    requireStage(StageName::Vertex, "Position()");
    checkType(value, vec4f(), "position");
    pushStatement(st::SetPosition{value.node()}, loc);
  }, value, loc);
}

void FragDepth(const ValueLike& value, Loc loc) {
  return detail::guard([&] {
    requireStage(StageName::Fragment, "FragDepth()");
    const Val v = toVal(value, f32());
    checkType(v, f32(), "FragDepth");
    pushStatement(st::SetFragDepth{v.node()}, loc);
  }, value, loc);
}

void ClipDistance(int index, const ValueLike& value, Loc loc) {
  return detail::guard([&] {
    requireStage(StageName::Vertex, "ClipDistance()");
    // WebGPU allows at most 8 clip distances
    if (index < 0 || index > 7)
      throw ShaderError(ErrorCode::OutOfRange, std::format("ClipDistance index {} is not in 0..7", index));
    const Val v = toVal(value, f32());
    checkType(v, f32(), "ClipDistance");
    pushStatement(st::SetClipDistance{index, v.node()}, loc);
  }, index, value, loc);
}

} // namespace moga
