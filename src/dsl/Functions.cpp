#include "Context.hpp"
#include "ControlFlow.hpp"
#include "moga/dsl/Functions.hpp"

#include <algorithm>
#include <format>
#include <set>
#include <string_view>

namespace moga {

using namespace detail;

const Val& Args::operator[](std::string_view name) const {
  for (std::size_t i = 0; i < names_.size(); ++i)
    if (names_[i] == name) return values_[i];
  fail(ShaderError(ErrorCode::UnknownName, std::format("No parameter \"{}\"", name)));
  // while recovering, a placeholder stands for the missing parameter
  static const Val missing = poison(f32());
  return missing;
}

namespace {

/** Function parameters and results are values; atomics and void are not. */
void checkSignatureType(const TypePtr& t, std::string_view function, std::string_view what) {
  if (containsAtomic(t))
    throw ShaderError(ErrorCode::InvalidType, std::format("Function {}: {} cannot be or hold an atomic ({})", function, what,
                                  typeName(t)));
}

} // namespace

Function fn(std::string name, const StructFields& params, TypePtr returnType,
            const std::function<void(const Args&)>& body, Loc loc) {
  detail::noteUserLocation(loc);
  auto decl = std::make_shared<FunctionDecl>();
  decl->name = std::move(name);
  decl->returnType = std::move(returnType);
  check([&] { checkSignatureType(decl->returnType, decl->name, "the return type"); });
  std::set<std::string_view> seen;
  for (const auto& [n, t] : params) {
    check([&] {
      // Args finds a parameter by name, so a repeated name would hide the later one
      if (!seen.insert(n).second)
        throw ShaderError(ErrorCode::DuplicateName, std::format("Function {}: parameter {} is declared twice", decl->name, n));
      if (t == voidType()) throw ShaderError(ErrorCode::InvalidType, std::format("Function {}: parameter {} is void", decl->name, n));
      checkSignatureType(t, decl->name, std::format("parameter {}", n));
    });
  }
  std::vector<std::string> names;
  std::vector<Val> values;
  for (const auto& [n, t] : params) {
    LocalPtr l = newLocal(n, t, false);
    decl->params.push_back(l);
    names.push_back(n);
    values.push_back(localVal(l));
  }
  const Args args(std::move(names), std::move(values));
  functionReturnTypes().push_back(decl->returnType);
  struct Pop {
    ~Pop() { functionReturnTypes().pop_back(); }
  } pop;
  const std::size_t errorsBefore = recordedErrorCount();
  decl->body = collectBody(Enclosing::Function, [&] { body(args); });
  // a path that reaches the end has not returned (WGSL behavior analysis); a
  // body cut short by an error would report a missing return that is not there
  const bool complete = recordedErrorCount() == errorsBefore;
  if (complete && decl->returnType != voidType() && (blockBehaviors(decl->body) & kNext) != 0)
    fail(ShaderError(ErrorCode::ControlFlow, std::format("Function {} can reach its end without returning a {}",
                                                         decl->name, typeName(decl->returnType))));
  return Function(std::move(decl));
}

Val Function::call(const std::vector<ValueLike>& args) const {
  return detail::guard([&]() -> Val {
    const FunctionDecl& f = *decl_;
    if (args.size() != f.params.size())
      throw ShaderError(ErrorCode::ArgumentCount, std::format("{}: expected {} arguments, got {}", f.name, f.params.size(),
                                    args.size()));
    std::vector<ExprPtr> callArgs;
    for (std::size_t i = 0; i < args.size(); ++i) {
      const LocalPtr& p = f.params[i];
      const Val v = toVal(args[i], p->type);
      checkType(v, p->type, std::format("{}({})", f.name, p->name));
      callArgs.push_back(v.node());
    }
    Val invocation(makeExpr(ex::UCall{decl_, std::move(callArgs)}, f.returnType));
    if (f.returnType == voidType()) pushStatement(st::ExprStmt{invocation.node()}, Loc::current());
    return invocation;
  }, args);
}

} // namespace moga
