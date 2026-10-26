#include "Context.hpp"
#include "Known.hpp"
#include "ControlFlow.hpp"
#include "moga/dsl/Statements.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <initializer_list>

namespace moga {

using namespace detail;

Val Var(const TypePtr& type, std::string name, Loc loc) {
  return detail::guard([&]() -> Val {
    requireValueType(type, std::format("Variable {}", name));
    LocalPtr v = newLocal(std::move(name), type, true);
    pushStatement(st::Decl{v, nullptr}, loc);
    return localVal(v);
  }, type, name, loc);
}

Val Var(const ValueLike& init, std::string name, Loc loc) {
  return detail::guard([&]() -> Val {
    const Val value = toVal(init);
    requireValueType(value.type(), std::format("Variable {}", name));
    LocalPtr v = newLocal(std::move(name), value.type(), true);
    pushStatement(st::Decl{v, value.node()}, loc);
    // a local made from a placeholder stands for it, so its uses report nothing more
    return poisoned(value) ? value : localVal(v);
  }, init, name, loc);
}

Val Let(const ValueLike& init, std::string name, Loc loc) {
  return detail::guard([&]() -> Val {
    const Val value = toVal(init);
    requireValueType(value.type(), std::format("Constant {}", name));
    LocalPtr v = newLocal(std::move(name), value.type(), false);
    pushStatement(st::Decl{v, value.node()}, loc);
    return poisoned(value) ? value : localVal(v);
  }, init, name, loc);
}

static Val condition(const ValueLike& cond, std::string_view what) {
  const Val c = detail::guard([&]() -> Val {
    const Val v = toVal(cond);
    if (v.type() != boolType())
      throw ShaderError(ErrorCode::TypeMismatch, std::format("{} condition must be bool, got {}", what, typeName(v.type())));
    return v;
  }, cond);
  // a placeholder condition is a bool, so the statement still takes it
  return detail::poisoned(c) ? detail::poison(boolType()) : c;
}

IfChain If(const ValueLike& cond, const std::function<void()>& body, Loc loc) {
  const Val c = condition(cond, "If");
  Block block = collectBlock(body);
  Stmt& s = pushStatement(st::If{c.node(), std::move(block), std::nullopt}, loc);
  return IfChain(&std::get<st::If>(s.node));
}

IfChain& IfChain::ElseIf(const ValueLike& cond, const std::function<void()>& body) {
  if (tail_->elseBody) {
    fail(ShaderError(ErrorCode::ControlFlow, "ElseIf() after Else()"));
    return *this;
  }
  const Val c = condition(cond, "ElseIf");
  auto nested = std::make_shared<Stmt>(Stmt{st::If{c.node(), collectBlock(body), std::nullopt}, {}});
  tail_->elseBody = Block{nested};
  tail_ = &std::get<st::If>(nested->node);
  return *this;
}

void IfChain::Else(const std::function<void()>& body) {
  return detail::guard([&] {
    if (tail_->elseBody) throw ShaderError(ErrorCode::ControlFlow, "Else() is already set for this If");
    tail_->elseBody = collectBlock(body);
  }, body);
}

void For(const ValueLike& start, const ValueLike& end, const std::function<void(const Val&)>& body,
         const ValueLike& step, Loc loc) {
  return detail::guard([&] {
    For(ForRange{start, end, step, BinaryOp::Lt}, body, loc);
  }, start, end, body, step, loc);
}

namespace {

/**
 * With known bounds, a counter that steps past the largest (or smallest)
 * value of its type wraps around before the condition stops it: WGSL integer
 * addition wraps, so the loop never ends.
 */
void requireNoWrapAt(double first, double last, double by, BinaryOp cond, bool isUnsigned);

/** Checks every combination of the ends of the possible start, end and step values. */
void requireNoWrap(const Val& start, const Val& end, const Val& step, BinaryOp cond) {
  const auto s = knownRanges(*start.node());
  const auto e = knownRanges(*end.node());
  const auto d = knownRanges(*step.node());
  if (!s || !e || !d) return;
  const bool isUnsigned = start.type()->scalar == ScalarKind::U32;
  for (const double first : {s->front().low, s->front().high})
    for (const double last : {e->front().low, e->front().high})
      for (const double by : {d->front().low, d->front().high})
        if (by != 0) requireNoWrapAt(first, last, by, cond, isUnsigned);
}

void requireNoWrapAt(double first, double last, double by, BinaryOp cond, bool isUnsigned) {
  const double lowest = isUnsigned ? 0.0 : kI32Min;
  const double highest = isUnsigned ? kU32Max : kI32Max;
  const bool up = cond == BinaryOp::Lt || cond == BinaryOp::Le;
  const auto runs = [&](double c) {
    if (cond == BinaryOp::Lt) return c < last;
    if (cond == BinaryOp::Le) return c <= last;
    if (cond == BinaryOp::Gt) return c > last;
    return c >= last;
  };
  if (!runs(first)) return;
  // the last counter value that runs, then the one after it
  const double passes = up ? std::floor((last - first) / by) : std::floor((first - last) / -by);
  double counter = first + passes * by;
  if (!runs(counter)) counter -= by;
  const double next = counter + by;
  if (next > highest || next < lowest)
    throw ShaderError(ErrorCode::ControlFlow, std::format("For: the counter wraps around after {} instead of reaching the end {}", counter, last));
}

/** A step known at build time that does not move the counter toward the end loops for ever. */
void requireStepTowardEnd(const Val& step, BinaryOp cond) {
  const auto known = knownRanges(*step.node());
  if (!known) return;
  const KnownRange& n = known->front();
  const bool up = cond == BinaryOp::Lt || cond == BinaryOp::Le;
  if ((up && n.low <= 0) || (!up && n.high >= 0))
    throw ShaderError(ErrorCode::ControlFlow, std::format("For: a step of {} never reaches the end with \"{}\"", describe(n), opText(cond)));
}

} // namespace

void For(const ForRange& range, const std::function<void(const Val&)>& body, Loc loc) {
  return detail::guard([&] {
    const BinaryOp cond = range.condition;
    if (cond != BinaryOp::Lt && cond != BinaryOp::Le && cond != BinaryOp::Gt && cond != BinaryOp::Ge)
      throw ShaderError(ErrorCode::ControlFlow, "For condition must be Lt, Le, Gt or Ge");
    // Counter type: an explicit Val start wins, then a Val end, else i32.
    TypePtr counterType = i32();
    if (range.start.val()) counterType = range.start.val()->type();
    else if (range.end.val()) counterType = range.end.val()->type();
    const Val s = toVal(range.start, counterType);
    if (s.type() != i32() && s.type() != u32()) throw ShaderError(ErrorCode::TypeMismatch, "For counter must be i32 or u32");
    const Val e = toVal(range.end, s.type());
    if (const double* n = range.step.number(); n != nullptr && *n < 0 && s.type() == u32())
      throw ShaderError(ErrorCode::ControlFlow, "For: a u32 counter cannot have a negative step; use an i32 counter");
    const Val stp = toVal(range.step, s.type());
    if (e.type() != s.type() || stp.type() != s.type())
      throw ShaderError(ErrorCode::TypeMismatch, "For bounds must share the counter type");
    requireStepTowardEnd(stp, cond);
    requireNoWrap(s, e, stp, cond);
    LocalPtr v = newLocal("i", s.type(), false);
    Block bodyStmts = collectBody(Enclosing::Loop, [&] { body(localVal(v)); });
    pushStatement(st::For{v, s.node(), e.node(), stp.node(), std::move(bodyStmts), cond}, loc);
  }, range, body, loc);
}

void ForGrid(const ValueLike& rows, const ValueLike& cols,
             const std::function<void(const Val&, const Val&)>& body, Loc loc) {
  return detail::guard([&] {
    For(0, rows, [&](const Val& i) { For(0, cols, [&](const Val& j) { body(i, j); }, 1, loc); }, 1,
        loc);
  }, rows, cols, body, loc);
}

void While(const ValueLike& cond, const std::function<void()>& body, Loc loc) {
  return detail::guard([&] {
    const Val c = condition(cond, "While");
    pushStatement(st::While{c.node(), collectBody(Enclosing::Loop, body)}, loc);
  }, cond, body, loc);
}

void Loop(const std::function<void()>& body, Loc loc) {
  return detail::guard([&] {
    const std::size_t errorsBefore = recordedErrorCount();
    Block block = collectBody(Enclosing::Loop, body);
    // WGSL rejects a loop that cannot exit; a body cut short by an error may have lost its Break()
    const bool complete = recordedErrorCount() == errorsBefore;
    if (complete && (blockBehaviors(block) & (kBreak | kReturn)) == 0)
      throw ShaderError(ErrorCode::ControlFlow, "Loop() never exits; add a Break() or a Return()");
    pushStatement(st::Loop{std::move(block)}, loc);
  }, body, loc);
}

SwitchChain Switch(const ValueLike& selector, Loc loc) {
  const Val s = detail::guard([&]() -> Val { return toVal(selector, i32()); }, selector);
  if (s.type() != i32() && s.type() != u32() && !detail::poisoned(s))
    fail(ShaderError(ErrorCode::TypeMismatch, std::format("Switch selector must be i32 or u32, got {}", typeName(s.type()))));
  Stmt& stmt = pushStatement(st::Switch{s.node(), {}, std::nullopt}, loc);
  return SwitchChain(&std::get<st::Switch>(stmt.node));
}

SwitchChain& SwitchChain::Case(const std::vector<ValueLike>& values,
                               const std::function<void()>& body) {
  if (values.empty()) fail(ShaderError(ErrorCode::ArgumentCount, "Switch case needs at least one value"));
  const TypePtr& type = switch_->selector->type;
  st::SwitchCase c;
  // every label is a literal; compare the converted values so 1 and 1.0 count as the same
  const auto literalOf = [](const ExprPtr& e) { return std::get<double>(std::get<ex::Lit>(e->node).value); };
  const auto used = [&](double value) {
    for (const st::SwitchCase& earlier : switch_->cases)
      for (const ExprPtr& e : earlier.values)
        if (literalOf(e) == value) return true;
    for (const ExprPtr& e : c.values)
      if (literalOf(e) == value) return true;
    return false;
  };
  for (const ValueLike& v : values) {
    // a value that fails is left out; the case keeps its other values and its body
    if (v.number() == nullptr) {
      fail(ShaderError(ErrorCode::ConstantRequired, "Switch case values must be number literals"));
      continue;
    }
    const Val label = detail::guard([&]() -> Val { return toVal(v, type); }, v);
    if (detail::poisoned(label)) continue;
    if (used(literalOf(label.node()))) {
      fail(ShaderError(ErrorCode::DuplicateName, std::format("Switch case value {} is used twice", literalOf(label.node()))));
      continue;
    }
    c.values.push_back(label.node());
  }
  c.body = collectBody(Enclosing::Switch, body);
  switch_->cases.push_back(std::move(c));
  return *this;
}

SwitchChain& SwitchChain::Case(const ValueLike& value, const std::function<void()>& body) {
  return Case(std::vector<ValueLike>{value}, body);
}

void SwitchChain::Default(const std::function<void()>& body) {
  return detail::guard([&] {
    if (switch_->defaultBody) throw ShaderError(ErrorCode::ControlFlow, "Default() is already set for this Switch");
    switch_->defaultBody = collectBody(Enclosing::Switch, body);
  }, body);
}

namespace {

/** Returns true when `kind` encloses the current block inside the current function body. */
bool insideOf(std::initializer_list<Enclosing> kinds) {
  const std::vector<Enclosing>& stack = enclosing();
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    if (*it == Enclosing::Function) return false;
    if (std::ranges::find(kinds, *it) != kinds.end()) return true;
  }
  return false;
}

} // namespace

void Break(Loc loc) {
  return detail::guard([&] {
    if (!insideOf({Enclosing::Loop, Enclosing::Switch}))
      throw ShaderError(ErrorCode::ControlFlow, "Break() is only valid in a loop or a switch case");
    pushStatement(st::Break{}, loc);
  }, loc);
}

void Continue(Loc loc) {
  return detail::guard([&] {
    // a switch inside a loop does not stop Continue(); both backends accept it
    if (!insideOf({Enclosing::Loop})) throw ShaderError(ErrorCode::ControlFlow, "Continue() is only valid in a loop");
    pushStatement(st::Continue{}, loc);
  }, loc);
}
void Return(Loc loc) {
  return detail::guard([&] {
    const std::vector<TypePtr>& open = functionReturnTypes();
    if (!open.empty() && open.back() != voidType())
      throw ShaderError(ErrorCode::TypeMismatch, std::format("Return() needs a {} value", typeName(open.back())));
    pushStatement(st::Return{nullptr}, loc);
  }, loc);
}

void Return(const ValueLike& value, Loc loc) {
  return detail::guard([&] {
    const std::vector<TypePtr>& open = functionReturnTypes();
    if (open.empty()) throw ShaderError(ErrorCode::ControlFlow, "A stage body cannot return a value; use Return()");
    const TypePtr& want = open.back();
    if (want == voidType()) throw ShaderError(ErrorCode::ControlFlow, "A void function cannot return a value");
    const Val v = toVal(value, want);
    checkType(v, want, "Return value");
    pushStatement(st::Return{v.node()}, loc);
  }, value, loc);
}

void Discard(Loc loc) {
  return detail::guard([&] {
    if (!buildingDetachedFunction()) requireStage(StageName::Fragment, "Discard()");
    pushStatement(st::Discard{}, loc);
  }, loc);
}

} // namespace moga
