#include "ControlFlow.hpp"

namespace moga::detail {

namespace {

/** A loop leaves through Break (then runs the next statement) or Return; Continue repeats it. */
Behaviors loopBehaviors(Behaviors body) {
  if ((body & kBreak) != 0) return (body & ~(kBreak | kContinue)) | kNext;
  return body & ~(kNext | kContinue);
}

/** Break in a switch case leaves only the switch. */
Behaviors caseBehaviors(Behaviors c) { return (c & kBreak) != 0 ? (c & ~kBreak) | kNext : c; }

Behaviors switchBehaviors(const st::Switch& sw, Marker marker) {
  Behaviors all = sw.defaultBody ? caseBehaviors(blockBehaviors(*sw.defaultBody, marker)) : kNext;
  for (const st::SwitchCase& c : sw.cases) all |= caseBehaviors(blockBehaviors(c.body, marker));
  return all;
}

} // namespace

Behaviors statementBehaviors(const Stmt& s, Marker marker) {
  if (marker != nullptr && marker(s)) return 0;
  return std::visit(
      Overloaded{
          [](const st::Return&) -> Behaviors { return kReturn; },
          [](const st::Break&) -> Behaviors { return kBreak; },
          [](const st::Continue&) -> Behaviors { return kContinue; },
          [&](const st::If& i) -> Behaviors {
            return blockBehaviors(i.body, marker) |
                   (i.elseBody ? blockBehaviors(*i.elseBody, marker) : kNext);
          },
          [&](const st::Switch& sw) { return switchBehaviors(sw, marker); },
          [&](const st::Loop& l) { return loopBehaviors(blockBehaviors(l.body, marker)); },
          // For and While test their condition first, which is a Break: they
          // can always reach the next statement, even with zero iterations
          [&](const st::For& f) { return loopBehaviors(blockBehaviors(f.body, marker) | kBreak); },
          [&](const st::While& w) { return loopBehaviors(blockBehaviors(w.body, marker) | kBreak); },
          [](const auto&) -> Behaviors { return kNext; },
      },
      s.node);
}

Behaviors blockBehaviors(const Block& block, Marker marker) {
  Behaviors left = 0;
  for (const StmtPtr& s : block) {
    const Behaviors b = statementBehaviors(*s, marker);
    left |= b & ~kNext;
    // no path reaches the next statement: the rest of the block never runs
    if ((b & kNext) == 0) return left;
  }
  return left | kNext;
}

} // namespace moga::detail
