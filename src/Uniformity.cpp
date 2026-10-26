#include "Uniformity.hpp"

#include "BuiltinTable.hpp"
#include "Walk.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace moga::detail {

namespace {

/** Derivatives and subgroup operations combine the values of other invocations, so they all must take part. */
bool needsUniformFlow(BuiltinFunction fn) {
  return fn == BuiltinFunction::Dpdx || fn == BuiltinFunction::Dpdy || fn == BuiltinFunction::Fwidth ||
         builtinSpec(fn).subgroup;
}

/**
 * Leaves whose value is the same for every invocation. The subgroup size is
 * uniform in a compute dispatch, but not across the fragments of a draw
 * (Tint, Chrome 155).
 */
bool uniformLeaf(const Expr& e, bool compute) {
  if (const auto* b = std::get_if<ex::Builtin>(&e.node))
    return b->name == BuiltinInput::WorkgroupId || b->name == BuiltinInput::NumWorkgroups ||
           (compute && b->name == BuiltinInput::SubgroupSize);
  return std::holds_alternative<ex::Lit>(e.node) || std::holds_alternative<ex::Constant>(e.node) ||
         std::holds_alternative<ex::Uniform>(e.node) || std::holds_alternative<ex::LooseUniform>(e.node) ||
         std::holds_alternative<ex::StorageLength>(e.node) || std::holds_alternative<ex::StorageSize>(e.node);
}

/** Reads of memory that other invocations can write, and per invocation inputs. */
bool varyingSource(const Expr& e) {
  if (const auto* s = std::get_if<ex::StorageElement>(&e.node)) return s->buffer->access != StorageAccess::Read;
  if (const auto* l = std::get_if<ex::StorageLoad>(&e.node)) return l->tex->access != StorageAccess::Read;
  // derivatives and subgroup operations combine invocations, so their result differs between them
  // even for a uniform argument (Tint: if (dpdx(u.x) > 0.0) does not guard a derivative)
  if (const auto* c = std::get_if<ex::Call>(&e.node)) return needsUniformFlow(c->fn);
  return std::holds_alternative<ex::Builtin>(e.node) || std::holds_alternative<ex::Attribute>(e.node) ||
         std::holds_alternative<ex::Varying>(e.node) || std::holds_alternative<ex::Workgroup>(e.node) ||
         std::holds_alternative<ex::Atomic>(e.node);
}

/** The local that an assignment target writes, or null for storage and workgroup memory. */
const LocalVar* targetLocal(const Expr& e) {
  if (const auto* l = std::get_if<ex::Local>(&e.node)) return l->v.get();
  if (const auto* m = std::get_if<ex::Member>(&e.node)) return targetLocal(*m->base);
  if (const auto* s = std::get_if<ex::Swizzle>(&e.node)) return targetLocal(*s->base);
  if (const auto* i = std::get_if<ex::Index>(&e.node)) return targetLocal(*i->base);
  return nullptr;
}

class Checker;

/** One body: a stage, or a function for one pattern of uniform arguments and flow. */
struct Body {
  std::set<const LocalVar*> tainted;
  std::unordered_map<const Expr*, bool> uniform;
  bool returnedUnevenly = false;
  bool reportRequirements = false;
};

class Checker {
public:
  Checker(std::string_view program, bool compute, Errors& errors)
      : program_(program), compute_(compute), errors_(errors) {}

  /** Analyzes a body until the tainted locals stop changing, then checks its barriers and derivatives. */
  bool run(const Block& block, Body& body, bool flow) {
    std::size_t before = 0;
    do {
      before = body.tainted.size();
      body.uniform.clear();
      body.returnedUnevenly = false;
      walk(block, body, flow, nullptr);
    } while (body.tainted.size() != before);
    body.uniform.clear();
    body.returnedUnevenly = false;
    body.reportRequirements = true;
    walk(block, body, flow, nullptr);
    return !body.returnedUnevenly;
  }

  /** Whether a call returns a uniform value, for this pattern of uniform arguments and flow. */
  bool call(const ex::UCall& c, Body& caller, bool flow) {
    std::vector<bool> args;
    for (const ExprPtr& a : c.args) args.push_back(isUniform(*a, caller));
    const auto key = std::tuple(c.fn.get(), args, flow);
    if (const auto it = calls_.find(key); it != calls_.end()) return it->second;
    calls_[key] = false;
    Body body;
    for (std::size_t i = 0; i < args.size(); ++i)
      if (!args[i]) body.tainted.insert(c.fn->params[i].get());
    // a call inside the callee resets this too, so the caller's value is kept aside
    const bool outer = returnsUniform_;
    returnsUniform_ = true;
    run(c.fn->body, body, flow);
    // a non uniform argument only matters where it reaches a return value or a return branch,
    // and its parameter is tainted for exactly that
    const bool result = returnsUniform_;
    returnsUniform_ = outer;
    calls_[key] = result;
    return result;
  }

  bool isUniform(const Expr& e, Body& body) {
    if (const auto it = body.uniform.find(&e); it != body.uniform.end()) return it->second;
    bool result = false;
    if (const auto* l = std::get_if<ex::Local>(&e.node)) result = !body.tainted.contains(l->v.get());
    else if (uniformLeaf(e, compute_)) result = true;
    else if (varyingSource(e)) result = false;
    else if (const auto* c = std::get_if<ex::UCall>(&e.node)) result = call(*c, body, true);
    else result = std::ranges::all_of(exprOperands(e), [&](const ExprPtr& o) { return isUniform(*o, body); });
    body.uniform.emplace(&e, result);
    return result;
  }

private:
  /** A loop body or a switch case. A switch case ends with break, but continue belongs to its loop. */
  struct Loop {
    bool uneven = false;
    Loop* outer = nullptr;
    bool switchCase = false;
  };

  static bool unevenChain(const Loop* loop) {
    for (; loop != nullptr; loop = loop->outer)
      if (loop->uneven) return true;
    return false;
  }

  void walk(const Block& block, Body& body, bool flow, Loop* loop) {
    for (const StmtPtr& s : block) stmt(*s, body, flow && !body.returnedUnevenly && !unevenChain(loop), loop);
  }

  void stmt(const Stmt& s, Body& body, bool flow, Loop* loop) {
    // a loop header runs again after each pass; loopHeader() checks it with the body known
    if (!std::holds_alternative<st::For>(s.node) && !std::holds_alternative<st::While>(s.node))
      requirements(s, body, flow);
    taint(s, body, flow);
    std::visit(Overloaded{
                   [&](const st::If& x) { branches(s, isUniform(*x.cond, body), body, flow, loop); },
                   [&](const st::Switch& x) { branches(s, isUniform(*x.selector, body), body, flow, loop); },
                   [&](const st::For& x) {
                     const bool bounds = isUniform(*x.start, body) && isUniform(*x.end, body) && isUniform(*x.step, body);
                     requirementsOf(*x.start, body, flow);
                     const bool even = loopBody(x.body, body, flow && bounds);
                     requirementsOf(*x.end, body, flow && bounds && even);
                     requirementsOf(*x.step, body, flow && bounds && even);
                   },
                   [&](const st::While& x) {
                     const bool uniformCondition = isUniform(*x.cond, body);
                     const bool even = loopBody(x.body, body, flow && uniformCondition);
                     requirementsOf(*x.cond, body, flow && uniformCondition && even);
                   },
                   [&](const st::Loop& x) { loopBody(x.body, body, flow); },
                   [&](const st::Return& x) {
                     // which return an invocation takes decides the result too
                     if (x.value && (!flow || !isUniform(*x.value, body))) returnsUniform_ = false;
                     if (!flow) body.returnedUnevenly = true;
                   },
                   [&](const st::Break&) { jump(loop, flow, false); },
                   [&](const st::Continue&) { jump(loop, flow, true); },
                   [&](const st::Barrier& b) { require(body, flow, barrierName(b.kind)); },
                   [](const auto&) {},
               },
               s.node);
  }

  void branches(const Stmt& s, bool uniformCondition, Body& body, bool flow, Loop* loop) {
    // a break in a switch case leaves the switch, not the loop
    const bool isSwitch = std::holds_alternative<st::Switch>(s.node);
    for (const Block* b : nestedBlocks(s)) {
      Loop caseLoop{false, loop, true};
      walk(*b, body, flow && uniformCondition, isSwitch ? &caseLoop : loop);
    }
  }

  /**
   * A break or continue that not every invocation takes makes the rest of the
   * loop uneven, on every pass. Returns false then: the header of the next pass
   * runs for only some invocations, as it does after a non uniform condition.
   */
  bool loopBody(const Block& block, Body& body, bool flow) {
    // an invocation that returns from the body skips the next header too
    const bool returnedBefore = body.returnedUnevenly;
    Loop loop;
    walk(block, body, flow, &loop);
    if (loop.uneven) walk(block, body, false, &loop);
    return !loop.uneven && (returnedBefore || !body.returnedUnevenly);
  }

  /** An uneven break ends the rest of its case or loop unevenly; an uneven continue, the rest of its loop. */
  static void jump(Loop* loop, bool flow, bool isContinue) {
    while (isContinue && loop != nullptr && loop->switchCase) loop = loop->outer;
    if (loop != nullptr && !flow) loop->uneven = true;
  }

  static std::string barrierName(BarrierKind kind) {
    static constexpr std::array<std::string_view, 3> kNames{"workgroupBarrier()", "storageBarrier()",
                                                            "textureBarrier()"};
    return std::string(kNames[static_cast<std::size_t>(kind)]);
  }

  /** Locals assigned a non uniform value, or assigned in non uniform flow. */
  void taint(const Stmt& s, Body& body, bool flow) {
    const auto mark = [&](const LocalVar* v, bool uniformValue) {
      if (v != nullptr && (!flow || !uniformValue)) body.tainted.insert(v);
    };
    if (const auto* d = std::get_if<st::Decl>(&s.node)) mark(d->v.get(), !d->init || isUniform(*d->init, body));
    if (const auto* a = std::get_if<st::Assign>(&s.node))
      mark(targetLocal(*a->target), isUniform(*a->value, body) && isUniform(*a->target, body));
    if (const auto* f = std::get_if<st::For>(&s.node))
      mark(f->v.get(), isUniform(*f->start, body) && isUniform(*f->end, body) && isUniform(*f->step, body));
  }

  /** Derivatives and calls in the statement's own expressions. */
  void requirements(const Stmt& s, Body& body, bool flow) {
    std::set<std::pair<const Expr*, bool>> seen;
    for (const ExprPtr& e : stmtOperands(s)) requirementsIn(*e, body, flow, seen);
  }

  void requirementsOf(const Expr& e, Body& body, bool flow) {
    std::set<std::pair<const Expr*, bool>> seen;
    requirementsIn(e, body, flow, seen);
  }

  /** The right operand of && and || runs only for some invocations unless the left one is uniform. */
  void requirementsIn(const Expr& e, Body& body, bool flow, std::set<std::pair<const Expr*, bool>>& seen) {
    if (!seen.insert({&e, flow}).second || std::holds_alternative<ex::Constant>(e.node)) return;
    const auto* b = std::get_if<ex::Binary>(&e.node);
    if (b != nullptr && (b->op == BinaryOp::And || b->op == BinaryOp::Or)) {
      requirementsIn(*b->a, body, flow, seen);
      requirementsIn(*b->b, body, flow && isUniform(*b->a, body), seen);
    } else {
      for (const ExprPtr& o : exprOperands(e)) requirementsIn(*o, body, flow, seen);
    }
    if (const auto* c = std::get_if<ex::Call>(&e.node); c != nullptr && needsUniformFlow(c->fn))
      require(body, flow, std::format("{}()", builtinName(c->fn)));
    if (const auto* u = std::get_if<ex::UCall>(&e.node); u != nullptr && body.reportRequirements) call(*u, body, flow);
  }

  void require(const Body& body, bool flow, const std::string& what) const {
    if (!body.reportRequirements || flow) return;
    std::string message = std::format("{}: {} must be reached by all invocations together, but a condition, loop "
                                      "bound or early exit before it can differ between them",
                                      program_, what);
    // a function walked for several call patterns can report the same requirement again
    if (std::ranges::none_of(errors_, [&](const ShaderError& e) { return e.message() == message; }))
      errors_.emplace_back(ErrorCode::Uniformity, std::move(message));
  }

  std::string_view program_;
  bool compute_;
  Errors& errors_;
  std::map<std::tuple<const FunctionDecl*, std::vector<bool>, bool>, bool> calls_;
  bool returnsUniform_ = true;
};

} // namespace

namespace {

/** Whether a block, or a function it calls, has a barrier or a derivative. */
class Needs {
public:
  bool block(const Block& b) {
    return std::ranges::any_of(b, [&](const StmtPtr& s) { return stmt(*s); });
  }

private:
  bool stmt(const Stmt& s) {
    if (std::holds_alternative<st::Barrier>(s.node)) return true;
    for (const ExprPtr& e : stmtOperands(s))
      if (e->needsUniformity && expr(*e)) return true;
    return std::ranges::any_of(nestedBlocks(s), [&](const Block* b) { return block(*b); });
  }

  bool expr(const Expr& e) {
    if (!e.needsUniformity || !seen_.insert(&e).second) return false;
    if (const auto* c = std::get_if<ex::Call>(&e.node); c != nullptr && needsUniformFlow(c->fn)) return true;
    if (const auto* u = std::get_if<ex::UCall>(&e.node); u != nullptr && function(*u->fn)) return true;
    return std::ranges::any_of(exprOperands(e), [&](const ExprPtr& o) { return expr(*o); });
  }

  bool function(const FunctionDecl& f) {
    if (const auto it = functions_.find(&f); it != functions_.end()) return it->second;
    functions_[&f] = false;
    const bool result = block(f.body);
    functions_[&f] = result;
    return result;
  }

  std::set<const Expr*> seen_;
  std::map<const FunctionDecl*, bool> functions_;
};

} // namespace

void checkUniformity(const ProgramAST& ast, Errors& errors) {
  // most programs have neither barriers nor derivatives
  if (!Needs().block(ast.compute ? ast.computeBody : ast.fragment)) return;
  Checker checker(ast.name, ast.compute, errors);
  Body compute;
  if (ast.compute) checker.run(ast.computeBody, compute, true);
  Body fragment;
  checker.run(ast.fragment, fragment, true);
}

} // namespace moga::detail
