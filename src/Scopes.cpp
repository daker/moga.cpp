#include "Scopes.hpp"

#include "Errors.hpp"
#include "Walk.hpp"

#include <algorithm>
#include <format>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moga::detail {

namespace {

class ScopeChecker {
public:
  explicit ScopeChecker(Errors& errors) : errors_(errors) {}

  void block(const Block& b) {
    scopes_.emplace_back();
    for (const StmtPtr& s : b) stmt(*s);
    scopes_.pop_back();
  }

  void function(const FunctionDecl& f) {
    if (!checked_.insert(&f).second) return;
    // a function sees its parameters, never the locals of its caller
    std::vector<std::vector<const LocalVar*>> caller = std::exchange(scopes_, {});
    scopes_.emplace_back();
    for (const LocalPtr& p : f.params) scopes_.back().push_back(p.get());
    block(f.body);
    scopes_ = std::move(caller);
  }

private:
  void stmt(const Stmt& s) {
    if (const auto* f = std::get_if<st::For>(&s.node)) {
      // the start is read before the loop variable exists; the bounds and the body see it
      read(*f->start, s);
      scopes_.push_back({f->v.get()});
      read(*f->end, s);
      read(*f->step, s);
      block(f->body);
      scopes_.pop_back();
      return;
    }
    for (const ExprPtr& e : stmtOperands(s)) read(*e, s);
    if (const auto* d = std::get_if<st::Decl>(&s.node)) scopes_.back().push_back(d->v.get());
    for (const Block* b : nestedBlocks(s)) block(*b);
  }

  /** Checks each local in the expression; each shared node once per statement. */
  void read(const Expr& root, const Stmt& s) {
    std::unordered_set<const Expr*> seen;
    const auto visit = [&](const Expr& e, auto&& self) -> void {
      // literals and named constants hold no local
      if (e.constantLeaves || !seen.insert(&e).second) return;
      if (const auto* l = std::get_if<ex::Local>(&e.node)) return requireVisible(*l->v, s);
      if (const auto* c = std::get_if<ex::UCall>(&e.node)) function(*c->fn);
      forEachOperand(e, [&](const Expr& operand) { self(operand, self); });
    };
    visit(root, visit);
  }

  void requireVisible(const LocalVar& v, const Stmt& s) const {
    const bool visible = std::ranges::any_of(
        scopes_, [&](const std::vector<const LocalVar*>& scope) { return std::ranges::contains(scope, &v); });
    if (visible) return;
    errors_.emplace_back(ErrorCode::Scope, std::format("Local {} is used outside the block that declares it", v.name),
                         s.debug);
  }

  Errors& errors_;
  std::vector<std::vector<const LocalVar*>> scopes_;
  std::unordered_set<const FunctionDecl*> checked_;
};

} // namespace

void checkLocalScopes(const ProgramAST& ast, Errors& errors) {
  ScopeChecker compute(errors);
  if (ast.compute) return compute.block(ast.computeBody);
  ScopeChecker vertex(errors);
  vertex.block(ast.vertex);
  ScopeChecker fragment(errors);
  fragment.block(ast.fragment);
}

} // namespace moga::detail
