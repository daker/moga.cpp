#include "Context.hpp"
#include "../Steps.hpp"
#include "moga/dsl/Program.hpp"

namespace moga {

using namespace detail;

ProgramAST program(std::string name, std::string variantKey, const std::function<void()>& vertex,
                   const std::function<void()>& fragment) {
  const auto prev = currentStage();
  struct Restore {
    std::optional<StageName> prev;
    ~Restore() { currentStage() = prev; }
  } restore{prev};
  ProgramAST ast{std::move(name), std::move(variantKey), {}, {}};
  currentStage() = StageName::Vertex;
  ast.vertex = collectBlock(vertex);
  currentStage() = StageName::Fragment;
  ast.fragment = collectBlock(fragment);
  return ast;
}

std::optional<StageName> inStage() { return currentStage(); }

} // namespace moga
