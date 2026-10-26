#include "Walk.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace moga::detail {

std::vector<ExprPtr> present(std::initializer_list<ExprPtr> list) {
  std::vector<ExprPtr> out;
  for (const ExprPtr& e : list)
    if (e) out.push_back(e);
  return out;
}

/** Expressions a statement evaluates directly, in source order (nested blocks excluded). */
std::vector<ExprPtr> stmtOperands(const Stmt& s) {
  return std::visit(
      Overloaded{
          [](const st::Decl& d) { return d.init ? std::vector{d.init} : std::vector<ExprPtr>{}; },
          [](const st::Assign& a) { return std::vector{a.target, a.value}; },
          [](const st::If& i) { return std::vector{i.cond}; },
          [](const st::For& f) { return std::vector{f.start, f.end, f.step}; },
          [](const st::While& w) { return std::vector{w.cond}; },
          [](const st::Switch& s) {
            std::vector<ExprPtr> out{s.selector};
            for (const st::SwitchCase& c : s.cases) out.insert(out.end(), c.values.begin(), c.values.end());
            return out;
          },
          [](const st::Return& r) {
            return r.value ? std::vector{r.value} : std::vector<ExprPtr>{};
          },
          [](const st::SetVarying& v) { return std::vector{v.value}; },
          [](const st::SetOutput& o) { return std::vector{o.value}; },
          [](const st::SetPosition& p) { return std::vector{p.value}; },
          [](const st::SetFragDepth& d) { return std::vector{d.value}; },
          [](const st::SetClipDistance& c) { return std::vector{c.value}; },
          [](const st::ExprStmt& e) { return std::vector{e.e}; },
          [](const st::AtomicStore& a) { return std::vector{a.target, a.value}; },
          [](const st::TextureStore& t) { return present({t.coord, t.layer, t.value}); },
          [](const auto&) { return std::vector<ExprPtr>{}; },
      },
      s.node);
}

std::vector<const Block*> nestedBlocks(const Stmt& s) {
  if (const auto* i = std::get_if<st::If>(&s.node)) {
    if (i->elseBody) return {&i->body, &*i->elseBody};
    return {&i->body};
  }
  if (const auto* f = std::get_if<st::For>(&s.node)) return {&f->body};
  if (const auto* l = std::get_if<st::Loop>(&s.node)) return {&l->body};
  if (const auto* w = std::get_if<st::While>(&s.node)) return {&w->body};
  if (const auto* sw = std::get_if<st::Switch>(&s.node)) {
    std::vector<const Block*> out;
    for (const st::SwitchCase& c : sw->cases) out.push_back(&c.body);
    if (sw->defaultBody) out.push_back(&*sw->defaultBody);
    return out;
  }
  return {};
}

/** Operands an expression reads, in source order. */
std::vector<ExprPtr> exprOperands(const Expr& e) {
  return std::visit(
      Overloaded{
          [](const ex::Constant& c) { return std::vector{c.c->value}; },
          [](const ex::Member& m) { return std::vector{m.base}; },
          [](const ex::Swizzle& s) { return std::vector{s.base}; },
          [](const ex::Index& i) { return std::vector{i.base, i.index}; },
          [](const ex::Unary& u) { return std::vector{u.a}; },
          [](const ex::Binary& b) { return std::vector{b.a, b.b}; },
          [](const ex::Call& c) { return c.args; },
          [](const ex::UCall& c) { return c.args; },
          [](const ex::Construct& c) { return c.args; },
          [](const ex::Select& s) { return std::vector{s.cond, s.t, s.f}; },
          [](const ex::Sample& s) {
            return present({s.coord, s.level, s.layer, s.ddx, s.ddy, s.offset, s.depthRef});
          },
          [](const ex::Load& l) { return present({l.coord, l.level, l.layer}); },
          [](const ex::TexSize& t) { return std::vector{t.level}; },
          [](const ex::TableRead& t) { return std::vector{t.index}; },
          [](const ex::StorageElement& s) { return std::vector{s.index}; },
          [](const ex::Atomic& a) { return present({a.target, a.value}); },
          [](const ex::StorageLoad& l) { return present({l.coord, l.layer}); },
          [](const auto&) { return std::vector<ExprPtr>{}; },
      },
      e.node);
}

namespace {

/** Statements whose expressions written out are at most this many nodes print as they are. */
constexpr std::uint64_t kSharedTreeSize = 64;

/** Nodes worth printing once: values, not references (an assignment target must stay one) or leaves. */
bool hoistable(const Expr& e) {
  if (hasSideEffects(e) || e.type->kind == TypeKind::Void) return false;
  return std::holds_alternative<ex::Binary>(e.node) || std::holds_alternative<ex::Unary>(e.node) ||
         std::holds_alternative<ex::Call>(e.node) || std::holds_alternative<ex::Construct>(e.node) ||
         std::holds_alternative<ex::Select>(e.node) || std::holds_alternative<ex::Sample>(e.node) ||
         std::holds_alternative<ex::Load>(e.node) || std::holds_alternative<ex::TableRead>(e.node);
}

struct SharedWalk {
  std::unordered_map<const Expr*, int> uses;
  std::unordered_set<const Expr*> visited;
  std::vector<const Expr*> order;

  void visit(const Expr& e) {
    ++uses[&e];
    if (!visited.insert(&e).second) return;
    // a named constant prints as its name
    if (!std::holds_alternative<ex::Constant>(e.node))
      for (const ExprPtr& operand : exprOperands(e)) visit(*operand);
    order.push_back(&e);
  }
};

} // namespace

std::vector<const Expr*> sharedSubexpressions(const Stmt& s) {
  if (std::holds_alternative<st::For>(s.node) || std::holds_alternative<st::While>(s.node)) return {};
  return sharedSubexpressions(stmtOperands(s));
}

std::vector<const Expr*> sharedSubexpressions(const ExprPtr& value) { return sharedSubexpressions(std::vector{value}); }

std::vector<const Expr*> sharedSubexpressions(const std::vector<ExprPtr>& roots) {
  // small expressions print as they are; sharing only matters where it can multiply the text
  std::uint64_t size = 0;
  for (const ExprPtr& root : roots) size += root->treeSize;
  if (size <= kSharedTreeSize) return {};
  SharedWalk walk;
  for (const ExprPtr& root : roots) walk.visit(*root);
  // a local is computed before the statement: a read of memory that a call or an atomic in the
  // statement writes must stay after that call, so it keeps its place
  const bool writes = std::ranges::any_of(roots, [](const ExprPtr& r) { return hasSideEffects(*r); });
  std::vector<const Expr*> shared;
  for (const Expr* e : walk.order)
    if (walk.uses[e] > 1 && hoistable(*e) && !(writes && e->readsMutableMemory)) shared.push_back(e);
  return shared;
}

} // namespace moga::detail
