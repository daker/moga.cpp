#pragma once

// Walks over statements and expressions, shared by analysis, construction
// and the backends.

#include "moga/Nodes.hpp"

#include <initializer_list>
#include <vector>

namespace moga::detail {

/** The non null expressions of the list. */
std::vector<ExprPtr> present(std::initializer_list<ExprPtr> list);

/** Expressions a statement evaluates directly, in source order (nested blocks excluded). */
std::vector<ExprPtr> stmtOperands(const Stmt& s);

/** The blocks nested in a statement. */
std::vector<const Block*> nestedBlocks(const Stmt& s);

/** Operands an expression reads, in source order; a named constant reads its value. */
std::vector<ExprPtr> exprOperands(const Expr& e);

/** Calls f on each operand that exprOperands() would return, without building a list. */
template <class F> void forEachOperand(const Expr& e, F&& f) {
  const auto each = [&](const ExprPtr& x) {
    if (x) f(*x);
  };
  std::visit(Overloaded{
                 [&](const ex::Constant& c) { each(c.c->value); },
                 [&](const ex::Member& m) { each(m.base); },
                 [&](const ex::Swizzle& s) { each(s.base); },
                 [&](const ex::Index& i) { each(i.base), each(i.index); },
                 [&](const ex::Unary& u) { each(u.a); },
                 [&](const ex::Binary& b) { each(b.a), each(b.b); },
                 [&](const ex::Call& c) { for (const ExprPtr& a : c.args) each(a); },
                 [&](const ex::UCall& c) { for (const ExprPtr& a : c.args) each(a); },
                 [&](const ex::Construct& c) { for (const ExprPtr& a : c.args) each(a); },
                 [&](const ex::Select& s) { each(s.cond), each(s.t), each(s.f); },
                 [&](const ex::Sample& s) {
                   for (const ExprPtr& a : {s.coord, s.level, s.layer, s.ddx, s.ddy, s.offset, s.depthRef}) each(a);
                 },
                 [&](const ex::Load& l) { each(l.coord), each(l.level), each(l.layer); },
                 [&](const ex::TexSize& t) { each(t.level); },
                 [&](const ex::TableRead& t) { each(t.index); },
                 [&](const ex::StorageElement& s) { each(s.index); },
                 [&](const ex::Atomic& a) { each(a.target), each(a.value); },
                 [&](const ex::StorageLoad& l) { each(l.coord), each(l.layer); },
                 [](const auto&) {},
             },
             e.node);
}

/**
 * The pure value nodes that a statement's expressions read more than once,
 * operands before the nodes that read them. Printing such a node once, as a
 * local, keeps an expression that reuses a value (x = x + x, n times) from
 * growing exponentially. Loop conditions and steps run on every iteration,
 * so a loop statement has none; nodes with side effects keep their place.
 */
std::vector<const Expr*> sharedSubexpressions(const Stmt& s);
/** The same for the value of a named constant, printed before it as more constants. */
std::vector<const Expr*> sharedSubexpressions(const ExprPtr& value);
/** The same for any expressions that are evaluated together. */
std::vector<const Expr*> sharedSubexpressions(const std::vector<ExprPtr>& roots);

} // namespace moga::detail
