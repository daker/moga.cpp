#include "moga/Analyze.hpp"
#include "Steps.hpp"

#include "ControlFlow.hpp"
#include "Errors.hpp"
#include "Lower.hpp"
#include "Scopes.hpp"
#include "Uniformity.hpp"
#include "Walk.hpp"
#include "BuiltinTable.hpp"
#include "TextEmitter.hpp"

#include <algorithm>
#include <format>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace moga {

using detail::exprOperands;
using detail::nestedBlocks;
using detail::present;
using detail::stmtOperands;

namespace {

template <class T, class U> void pushUnique(std::vector<T>& list, const U& item) {
  if (std::ranges::find(list, item) == list.end()) list.push_back(item);
}

/** The bound resource an expression reads, if any. */
ResourcePtr resourceOf(const Expr& e) {
  return std::visit(Overloaded{
                        [](const ex::Uniform& u) -> ResourcePtr { return u.r; },
                        [](const ex::Sample& s) -> ResourcePtr { return s.tex; },
                        [](const ex::Load& l) -> ResourcePtr { return l.tex; },
                        [](const ex::TexSize& t) -> ResourcePtr { return t.tex; },
                        [](const ex::TableRead& t) -> ResourcePtr { return t.table; },
                        [](const ex::StorageElement& s) -> ResourcePtr { return s.buffer; },
                        [](const ex::StorageLength& s) -> ResourcePtr { return s.buffer; },
                        [](const ex::StorageLoad& l) -> ResourcePtr { return l.tex; },
                        [](const ex::StorageSize& s) -> ResourcePtr { return s.tex; },
                        [](const auto&) -> ResourcePtr { return nullptr; },
                    },
                    e.node);
}

struct StageUsage {
  std::vector<AttributePtr> attributes;
  std::vector<VaryingPtr> varyingsWritten;
  std::vector<VaryingPtr> varyingsRead;
  std::vector<OutputPtr> outputs;
  std::vector<ResourcePtr> resources;
  std::vector<BuiltinInput> builtins;
  bool fragDepth = false;
  int clipDistances = 0;
  std::vector<FunctionPtr> functions;
  std::vector<ConstantPtr> constants;
  std::vector<UniformPtr> uniforms;
  /** Struct and array types of locals and expressions. */
  std::vector<TypePtr> compositeTypes;
  /** Helper functions that some backend needs. */
  std::vector<Helper> glslHelpers;
  std::vector<Helper> wgslHelpers;
  std::vector<WorkgroupPtr> workgroups;
  std::vector<DeviceFeature> deviceFeatures;
  std::vector<LanguageFeature> languageFeatures;
  std::vector<FeatureUse> features;
};

bool isComposite(const TypePtr& t) {
  return t->kind == TypeKind::Struct || t->kind == TypeKind::Array;
}

class UsageCollector {
public:
  UsageCollector(StageName stage, detail::Errors& errors) : stage_(stage), errors_(errors) {}

  StageUsage usage;

  void block(const Block& stmts, const FunctionDecl* inFunction) {
    for (const StmtPtr& s : stmts) stmt(*s, inFunction);
  }

private:
  /** Records an error at the statement being checked, and goes on. */
  void report(ErrorCode code, std::string message) const {
    errors_.emplace_back(code, std::move(message), at_ != nullptr ? at_->debug : std::nullopt);
  }

  void stageOnly(std::string_view what, const FunctionDecl* fn) const {
    if (fn != nullptr)
      report(ErrorCode::WrongStage,
             std::format("{} used inside function \"{}\"; stage IO is only valid in stage bodies", what, fn->name));
  }

  void stmt(const Stmt& s, const FunctionDecl* fn) {
    // a call checks the statements of the function; the rest of this statement is still here
    const Stmt* outer = std::exchange(at_, &s);
    recordStageOutput(s, fn);
    recordWebGPUStatement(s);
    if (const auto* d = std::get_if<st::Decl>(&s.node)) {
      if (isComposite(d->v->type)) pushUnique(usage.compositeTypes, d->v->type);
      recordHalf(d->v->type);
    }
    for (const ExprPtr& e : stmtOperands(s)) expr(*e, fn);
    for (const Block* b : nestedBlocks(s)) block(*b, fn);
    at_ = outer;
  }

  void recordStageOutput(const Stmt& s, const FunctionDecl* fn) {
    if (const auto* v = std::get_if<st::SetVarying>(&s.node)) {
      stageOnly(std::format("varying {}", v->v->name), fn);
      pushUnique(usage.varyingsWritten, v->v);
    } else if (const auto* o = std::get_if<st::SetOutput>(&s.node)) {
      stageOnly(std::format("output {}", o->o->index), fn);
      const auto same = std::ranges::find_if(usage.outputs, [&](const OutputPtr& w) {
        return w->index == o->o->index && w->blendSrc == o->o->blendSrc;
      });
      if (o->o->blendSrc) {
        use(Feature::DualSourceBlending, "dual source blending");
        pushUnique(usage.deviceFeatures, DeviceFeature::DualSourceBlending);
      }
      if (same == usage.outputs.end()) usage.outputs.push_back(o->o);
      else if (!typesEqual((*same)->type, o->o->type))
        report(ErrorCode::StageInterface, std::format("Output {} is written as {} and as {}", o->o->index,
                                                      typeName((*same)->type), typeName(o->o->type)));
    } else if (std::holds_alternative<st::SetPosition>(s.node)) {
      stageOnly("Position", fn);
    } else if (std::holds_alternative<st::SetFragDepth>(s.node)) {
      stageOnly("FragDepth", fn);
      usage.fragDepth = true;
    } else if (const auto* c = std::get_if<st::SetClipDistance>(&s.node)) {
      stageOnly("ClipDistance", fn);
      usage.clipDistances = std::max(usage.clipDistances, c->index + 1);
      pushUnique(usage.deviceFeatures, DeviceFeature::ClipDistances);
    }
  }

  void expr(const Expr& e, const FunctionDecl* fn) {
    // a reused value is a shared node: visiting it again would expand the graph into a tree. A
    // small node is revisited at most once per distinct parent, so only large ones need the set
    if (e.treeSize > kRevisitTreeSize && !visited_.insert({&e, fn}).second) return;
    recordStageInput(e, fn);
    if (ResourcePtr r = resourceOf(e)) useResource(r);
    recordWebGPUExpression(e);
    for (const ExprPtr& operand : exprOperands(e)) expr(*operand, fn);
    if (const auto* c = std::get_if<ex::UCall>(&e.node)) func(c->fn);
    // after its operands, so the constants that the value uses come first
    if (const auto* c = std::get_if<ex::Constant>(&e.node)) pushUnique(usage.constants, c->c);
    if (const auto* u = std::get_if<ex::LooseUniform>(&e.node)) pushUnique(usage.uniforms, u->u);
    if (isComposite(e.type)) pushUnique(usage.compositeTypes, e.type);
    recordHalf(e.type);
    recordHelper(e);
  }

  void recordHalf(const TypePtr& t) {
    if (!containsF16(t)) return;
    use(Feature::F16, "f16");
    pushUnique(usage.deviceFeatures, DeviceFeature::ShaderF16);
  }

  /** The helpers that each backend lowers the expression to. */
  void recordHelper(const Expr& e) {
    if (auto h = detail::lower(detail::Target::GLSL, e).helper) pushUnique(usage.glslHelpers, *h);
    if (auto h = detail::lower(detail::Target::WGSL, e).helper) pushUnique(usage.wgslHelpers, *h);
  }

  void recordStageInput(const Expr& e, const FunctionDecl* fn) {
    if (const auto* a = std::get_if<ex::Attribute>(&e.node)) {
      stageOnly(std::format("attribute {}", a->a->name), fn);
      if (stage_ != StageName::Vertex)
        return report(ErrorCode::WrongStage, std::format("Attribute {} read outside the vertex stage", a->a->name));
      pushUnique(usage.attributes, a->a);
    } else if (const auto* v = std::get_if<ex::Varying>(&e.node)) {
      stageOnly(std::format("varying {}", v->v->name), fn);
      pushUnique(usage.varyingsRead, v->v);
    } else if (const auto* b = std::get_if<ex::Builtin>(&e.node)) {
      stageOnly(std::format("builtin {}", builtinInputName(b->name)), fn);
      pushUnique(usage.builtins, b->name);
    }
  }

  void use(Feature f, std::string what) { pushUnique(usage.features, FeatureUse{f, std::move(what)}); }

  void useResource(const ResourcePtr& r) {
    pushUnique(usage.resources, r);
    if (r->kind == ResourceKind::StorageBuffer) {
      const auto& b = static_cast<const StorageBufferResource&>(*r);
      use(Feature::StorageBuffer, std::format("storage buffer {}", b.name));
      if (stage_ == StageName::Vertex && b.access == StorageAccess::ReadWrite)
        report(ErrorCode::WrongStage, std::format("Read/write storage {} is not valid in the vertex stage", b.name));
      if (isComposite(b.elementType)) pushUnique(usage.compositeTypes, b.elementType);
    } else if (r->kind == ResourceKind::StorageTexture) {
      const auto& t = static_cast<const StorageTextureResource&>(*r);
      use(Feature::StorageTexture, std::format("storage texture {}", t.name));
      if (stage_ == StageName::Vertex && t.access != StorageAccess::Read)
        report(ErrorCode::WrongStage, std::format("Writable storage texture {} is not valid in the vertex stage", t.name));
    }
  }

  void recordWebGPUStatement(const Stmt& s) {
    if (const auto* b = std::get_if<st::Barrier>(&s.node)) {
      static constexpr std::array<std::string_view, 3> names{"workgroupBarrier", "storageBarrier",
                                                             "textureBarrier"};
      use(Feature::Barrier, std::string(names[static_cast<int>(b->kind)]));
      // a helper function is built outside a stage; this is where its stage is known
      if (stage_ != StageName::Compute)
        report(ErrorCode::WrongStage, std::format("{}() is only valid in the compute stage", names[static_cast<int>(b->kind)]));
      if (b->kind == BarrierKind::Texture)
        pushUnique(usage.languageFeatures, LanguageFeature::ReadonlyAndReadwriteStorageTextures);
    } else if (std::holds_alternative<st::Discard>(s.node) && stage_ != StageName::Fragment) {
      report(ErrorCode::WrongStage, "Discard() is only valid in the fragment stage");
    } else if (std::holds_alternative<st::AtomicStore>(s.node)) {
      use(Feature::Atomics, "atomicStore");
    } else if (const auto* t = std::get_if<st::TextureStore>(&s.node)) {
      useResource(t->tex);
    }
  }

  void recordWebGPUExpression(const Expr& e) {
    if (const auto* w = std::get_if<ex::Workgroup>(&e.node)) {
      if (stage_ != StageName::Compute)
        report(ErrorCode::WrongStage, std::format("Workgroup array {} is only valid in the compute stage", w->w->name));
      use(Feature::WorkgroupArray, std::format("workgroup array {}", w->w->name));
      pushUnique(usage.workgroups, w->w);
    } else if (const auto* a = std::get_if<ex::Atomic>(&e.node)) {
      use(Feature::Atomics, std::string(atomicOpName(a->op)));
    } else if (const auto* b = std::get_if<ex::Builtin>(&e.node)) {
      if (isComputeBuiltin(b->name)) use(Feature::ComputeBuiltin, std::format("builtin {}", builtinInputName(b->name)));
      if (b->name == BuiltinInput::PrimitiveIndex) {
        use(Feature::PrimitiveIndex, "builtin primitiveIndex");
        pushUnique(usage.deviceFeatures, DeviceFeature::PrimitiveIndex);
      }
      if (isSubgroupBuiltin(b->name)) {
        pushUnique(usage.deviceFeatures, DeviceFeature::Subgroups);
        use(Feature::Subgroups, std::format("builtin {}", builtinInputName(b->name)));
      }
    } else if (const auto* d = std::get_if<ex::Call>(&e.node);
               d != nullptr && stage_ != StageName::Fragment &&
               detail::builtinSpec(d->fn).stages == detail::Stages::Fragment) {
      // a helper function is built outside a stage; this is where its stage is known
      report(ErrorCode::WrongStage, std::format("{} is only valid in the fragment stage", builtinName(d->fn)));
    } else if (const auto* c = std::get_if<ex::Call>(&e.node); c != nullptr && isSubgroupFunction(c->fn)) {
      if (stage_ == StageName::Vertex)
        report(ErrorCode::WrongStage, std::format("{} is only valid in the compute and fragment stages", builtinName(c->fn)));
      pushUnique(usage.deviceFeatures, DeviceFeature::Subgroups);
      use(Feature::Subgroups, std::string(builtinName(c->fn)));
    }
  }

  void func(const FunctionPtr& f) {
    if (std::ranges::find(usage.functions, f) != usage.functions.end()) return;
    if (visiting_.contains(f.get()))
      return report(ErrorCode::Recursion, std::format("Recursive shader function \"{}\"", f->name));
    visiting_.insert(f.get());
    block(f->body, f.get());
    visiting_.erase(f.get());
    // post order: callees are pushed before callers
    usage.functions.push_back(f);
  }

  StageName stage_;
  detail::Errors& errors_;
  /** The statement being checked, for the location of an error. */
  const Stmt* at_ = nullptr;
  std::set<const FunctionDecl*> visiting_;
  std::set<std::pair<const Expr*, const FunctionDecl*>> visited_;
  static constexpr std::uint32_t kRevisitTreeSize = 16;
};

void collectStructs(const std::vector<TypePtr>& types, std::vector<TypePtr>& out) {
  for (const TypePtr& t : types) {
    if (t->kind == TypeKind::Struct) {
      // the backends declare structs by name: one definition per name
      const auto same = std::ranges::find(out, t->name, &TypeInfo::name);
      if (same != out.end()) {
        if (!typesEqual(*same, t))
          throw ShaderError(ErrorCode::DuplicateName, std::format("Two different structs are named {}", t->name));
        continue;
      }
      std::vector<TypePtr> fieldTypes;
      for (const StructField& f : t->fields) fieldTypes.push_back(f.type);
      collectStructs(fieldTypes, out);
      out.push_back(t);
    } else if (t->kind == TypeKind::Array) {
      collectStructs({t->element}, out);
    }
  }
}

/** Every varying the fragment stage reads must be written by the vertex stage. */
void checkStageInterface(const StageUsage& vs, const StageUsage& fs, detail::Errors& errors) {
  for (const VaryingPtr& v : fs.varyingsRead)
    if (std::ranges::find(vs.varyingsWritten, v) == vs.varyingsWritten.end())
      errors.emplace_back(ErrorCode::StageInterface, std::format("Varying \"{}\" is read but never written", v->name));
  if (!vs.varyingsRead.empty()) errors.emplace_back(ErrorCode::WrongStage, "Varyings cannot be read in the vertex stage");
  // the backends and the reflection know a varying by its name; each pair once
  for (auto v = vs.varyingsWritten.begin(); v != vs.varyingsWritten.end(); ++v)
    if (std::any_of(std::next(v), vs.varyingsWritten.end(), [&](const VaryingPtr& w) { return (*v)->name == w->name; }))
      errors.emplace_back(ErrorCode::DuplicateName, std::format("Two different varyings are named \"{}\"", (*v)->name));
}

/**
 * The backends name each resource from its kind and its sanitized name, so
 * two distinct resources must not get the same shader name.
 */
void checkResourceNames(const std::vector<ResourcePtr>& resources) {
  std::set<std::string> names;
  for (const ResourcePtr& r : resources) {
    std::string_view prefix;
    switch (r->kind) {
    case ResourceKind::UniformBlock:
      prefix = "ub_";
      break;
    case ResourceKind::Table:
      prefix = "tb_";
      break;
    case ResourceKind::Texture:
      prefix = "t_";
      break;
    case ResourceKind::StorageBuffer:
      prefix = "sb_";
      break;
    case ResourceKind::StorageTexture:
      prefix = "st_";
      break;
    }
    const std::string emitted = std::string(prefix) + detail::sanitize(r->name);
    if (!names.insert(emitted).second)
      throw ShaderError(ErrorCode::DuplicateName, std::format("Two resources are emitted as {}; give them different names", emitted));
  }
}

/** Deduplicated resources sorted by bind group, then kind and name. */
std::vector<ResourcePtr> orderedResources(const std::vector<ResourcePtr>& resources) {
  std::vector<ResourcePtr> all;
  for (const ResourcePtr& r : resources) pushUnique(all, r);
  std::ranges::stable_sort(all, [](const ResourcePtr& a, const ResourcePtr& b) {
    const int ga = groupOfFrequency(a->frequency);
    const int gb = groupOfFrequency(b->frequency);
    if (ga != gb) return ga < gb;
    if (a->kind != b->kind) return a->kind < b->kind;
    return a->name < b->name;
  });
  return all;
}

/** Assigns WGSL bindings, GLSL uniform buffer points and texture units in order. */
/** WebGPU pairs unfilterable float and plain depth sampling with a non-filtering sampler. */
SamplerBinding samplerBindingOf(const TextureResource& t) {
  if (!t.sampled) return SamplerBinding::None;
  if (t.comparison) return SamplerBinding::Comparison;
  if (t.sampleType == TextureSampleType::Float) return SamplerBinding::Filtering;
  return SamplerBinding::NonFiltering;
}

void bindResources(const std::vector<ResourcePtr>& all, AnalyzedProgram& out) {
  std::array<int, 3> nextBinding{0, 0, 0};
  int nextUboPoint = 0;
  int nextUnit = 0;
  Reflection& r = out.reflection;
  for (const ResourcePtr& res : all) {
    const int group = groupOfFrequency(res->frequency);
    std::optional<ReflectedResource> reflected;
    switch (res->kind) {
    case ResourceKind::UniformBlock: {
      const auto& ub = static_cast<const UniformBlockResource&>(*res);
      ReflectedUniformBlock b{ub.name, ub.frequency, structLayout(ub.structType), group,
                              nextBinding[group]++, "UB_" + detail::sanitize(ub.name), nextUboPoint++};
      r.uniformBlocks.push_back(b);
      reflected = b;
      break;
    }
    case ResourceKind::Texture: {
      const auto& t = static_cast<const TextureResource&>(*res);
      const int binding = nextBinding[group]++;
      const int samplerBinding = t.sampled ? nextBinding[group]++ : -1;
      ReflectedTexture tex{t.name, t.dimension, t.sampleType, t.frequency, group, binding,
                           samplerBinding, "t_" + detail::sanitize(t.name), nextUnit++, samplerBindingOf(t)};
      r.textures.push_back(tex);
      reflected = tex;
      break;
    }
    case ResourceKind::Table: {
      const auto& t = static_cast<const TableResource&>(*res);
      ReflectedTable tb{t.name, t.frequency, t.stride, tableFieldOffsets(t.structType), group,
                        nextBinding[group]++, "tb_" + detail::sanitize(t.name), nextUnit++};
      r.tables.push_back(tb);
      reflected = tb;
      break;
    }
    case ResourceKind::StorageBuffer: {
      const auto& b = static_cast<const StorageBufferResource&>(*res);
      ReflectedStorageBuffer sb{b.name, b.frequency, b.access, b.elementType,
                                storageLayout(b.elementType), group, nextBinding[group]++};
      r.storageBuffers.push_back(sb);
      reflected = sb;
      break;
    }
    case ResourceKind::StorageTexture: {
      const auto& t = static_cast<const StorageTextureResource&>(*res);
      ReflectedStorageTexture st{t.name, t.frequency, t.dimension, t.format, t.access, group,
                                 nextBinding[group]++};
      r.storageTextures.push_back(st);
      reflected = st;
      break;
    }
    }
    out.resources.push_back({res, *reflected});
    r.groups[group].push_back(*reflected);
  }
}

std::string_view frequencyName(ResourceFrequency f) {
  switch (f) {
  case ResourceFrequency::Frame:
    return "frame";
  case ResourceFrequency::Object:
    return "object";
  case ResourceFrequency::Material:
    return "material";
  }
  return "?";
}

/** Packs the single uniforms of each frequency into one block, fields sorted by name. */
void packUniforms(const std::vector<UniformPtr>& used, AnalyzedProgram& out,
                  std::vector<ResourcePtr>& resources) {
  for (const ResourceFrequency f :
       {ResourceFrequency::Frame, ResourceFrequency::Object, ResourceFrequency::Material}) {
    std::vector<UniformPtr> group;
    for (const UniformPtr& u : used)
      if (u->frequency == f) group.push_back(u);
    if (group.empty()) continue;
    std::ranges::sort(group, {}, &UniformDecl::name);
    StructFields fields;
    for (std::size_t i = 0; i < group.size(); ++i) {
      if (i > 0 && group[i]->name == group[i - 1]->name)
        throw ShaderError(ErrorCode::DuplicateName, std::format("Two {} uniforms are named \"{}\"", frequencyName(f), group[i]->name));
      fields.emplace_back(group[i]->name, group[i]->type);
    }
    const std::string name = std::format("{}Uniforms", frequencyName(f));
    auto block = std::make_shared<const UniformBlockResource>(name, makeStruct(name + "Block", fields), f);
    for (const UniformPtr& u : group) out.uniforms.emplace_back(u, block);
    resources.push_back(block);
  }
}

/**
 * The backends declare functions, constants, structs, resources and workgroup
 * arrays at module scope under names built from sanitized text. Two distinct
 * declarations must not get the same name, and none may take a name that the
 * backends generate themselves.
 */
void checkModuleNames(const AnalyzedProgram& out) {
  // generated names; every name starting with moga_ is reserved as well
  std::set<std::string> names{"main", "vs_main", "fs_main", "cs_main", "VSIn", "VSOut", "FSIn",
                              "FSOut", "CSIn", "vin", "vout", "fin", "fout", "cin"};
  const auto add = [&](const std::string& name, std::string_view what) {
    if (name.starts_with("moga_"))
      throw ShaderError(ErrorCode::ReservedName, std::format("{} is emitted as {}; names starting with moga_ are reserved", what, name));
    if (!names.insert(name).second)
      throw ShaderError(ErrorCode::DuplicateName, std::format("{} is emitted as {}, which another declaration already uses",
                                    what, name));
  };
  for (const FunctionPtr& f : out.functions)
    add(detail::functionName(*f), std::format("Function {}", f->name));
  for (const ConstantPtr& c : out.constants)
    add(detail::constantName(c->name), std::format("Constant {}", c->name));
  for (const TypePtr& s : out.structs) add(detail::structName(s->name), std::format("Struct {}", s->name));
  for (const WorkgroupPtr& w : out.workgroups)
    add("wg_" + detail::sanitize(w->name), std::format("Workgroup array {}", w->name));
  for (const BoundResource& b : out.resources)
    if (const auto* t = std::get_if<ReflectedTexture>(&b.reflected); t != nullptr && t->samplerBinding >= 0)
      add("s_" + detail::sanitize(t->name), std::format("The sampler of {}", t->name));
  // resources, with the extra names that GLSL declares for them
  for (const BoundResource& b : out.resources) {
    const std::string name = detail::sanitize(b.decl->name);
    const std::string what = std::format("Resource {}", b.decl->name);
    switch (b.decl->kind) {
    case ResourceKind::UniformBlock:
      add("ub_" + name, what);
      add("UB_" + name, what);
      break;
    case ResourceKind::Table:
      add("tb_" + name, what);
      add("tb_" + name + "_fetch", what);
      add("tb_" + name + "_fetchMat4", what);
      break;
    case ResourceKind::Texture:
      add("t_" + name, what);
      break;
    case ResourceKind::StorageBuffer:
      add("sb_" + name, what);
      break;
    case ResourceKind::StorageTexture:
      add("st_" + name, what);
      break;
    }
  }
  // GLSL declares attributes, varyings and outputs as globals
  const Reflection& r = out.reflection;
  for (const ReflectedAttribute& a : r.attributes)
    add("a_" + detail::sanitize(a.name), std::format("Attribute {}", a.name));
  for (const ReflectedVarying& v : r.varyings)
    add("v_" + detail::sanitize(v.name), std::format("Varying {}", v.name));
  for (const ReflectedOutput& o : r.outputs) add(outputName(o.index, o.blendSrc), "An output");
}

/** True when the uniform and the storage layout of a struct place every field the same. */
bool sameLayouts(const TypePtr& s) {
  const StructLayout& u = *structLayout(s);
  const StorageLayout st = storageLayout(s);
  const auto same = [](const FieldLayout& a, const FieldLayout& b) {
    return a.offset == b.offset && a.size == b.size && a.align == b.align;
  };
  return u.size == st.size && u.align == st.align && std::ranges::equal(u.fields, st.fields, same);
}

/**
 * WGSL gives a uniform struct explicit @align/@size to match std140; a storage
 * buffer of the same struct would then get the uniform layout, not the
 * storage layout that reflection reports. Reject that sharing unless the two
 * layouts are the same. Nested structs are in both lists and checked on their own.
 */
void checkSharedStructs(const AnalyzedProgram& out) {
  std::vector<TypePtr> uniformStructs;
  for (const BoundResource& b : out.resources)
    if (b.decl->kind == ResourceKind::UniformBlock)
      collectStructs({static_cast<const UniformBlockResource&>(*b.decl).structType}, uniformStructs);
  std::vector<TypePtr> storageStructs;
  for (const ReflectedStorageBuffer& sb : out.reflection.storageBuffers)
    collectStructs({sb.elementType}, storageStructs);
  for (const TypePtr& s : storageStructs)
    if (std::ranges::find(uniformStructs, s->name, &TypeInfo::name) != uniformStructs.end() && !sameLayouts(s))
      throw ShaderError(ErrorCode::Layout, std::format(
          "Struct {} is used in a uniform block and in a storage buffer; their layouts differ", s->name));
}

} // namespace

const UniformBlockResource* AnalyzedProgram::blockOf(const UniformDecl* uniform) const {
  for (const auto& [u, block] : uniforms)
    if (u.get() == uniform) return block.get();
  return nullptr;
}

int groupOf(const ReflectedResource& r) {
  return std::visit([](const auto& x) { return x.group; }, r);
}

int bindingOf(const ReflectedResource& r) {
  return std::visit([](const auto& x) { return x.binding; }, r);
}

bool Reflection::usesVertexBuiltin(BuiltinInput b) const {
  return std::ranges::find(vertexBuiltins, b) != vertexBuiltins.end();
}

bool Reflection::usesFragmentBuiltin(BuiltinInput b) const {
  return std::ranges::find(fragmentBuiltins, b) != fragmentBuiltins.end();
}

bool Reflection::usesComputeBuiltin(BuiltinInput b) const {
  return std::ranges::find(computeBuiltins, b) != computeBuiltins.end();
}

bool Reflection::hasVarying(std::string_view name) const {
  return std::ranges::find(varyings, name, &ReflectedVarying::name) != varyings.end();
}

bool Reflection::needs(DeviceFeature f) const { return std::ranges::find(deviceFeatures, f) != deviceFeatures.end(); }

const ReflectedResource* AnalyzedProgram::find(const Resource* decl) const {
  for (const BoundResource& b : resources)
    if (b.decl.get() == decl) return &b.reflected;
  return nullptr;
}

namespace {

/**
 * Every path through the vertex body must write Position() before it ends or
 * returns: WGSL would return a zero position and GLSL an undefined one.
 */
void checkPositionWritten(const ProgramAST& ast, detail::Errors& errors) {
  // with the write as the marker, what is left are the paths without one
  const detail::Behaviors left = detail::blockBehaviors(
      ast.vertex, [](const Stmt& s) { return std::holds_alternative<st::SetPosition>(s.node); });
  if ((left & detail::kReturn) != 0)
    errors.emplace_back(ErrorCode::StageInterface,
                        std::format("{}: the vertex stage can return before Position() is written", ast.name));
  else if (left != 0)
    errors.emplace_back(ErrorCode::StageInterface,
                        std::format("{}: the vertex stage does not write Position() on every path", ast.name));
}

/**
 * Dual source blending has both blend sources at location 0, with one type,
 * and no other location output (checked with Tint in Chrome 155).
 */
void checkDualSource(const Reflection& r) {
  const auto blend = std::ranges::count_if(r.outputs, [](const ReflectedOutput& o) { return o.blendSrc.has_value(); });
  if (blend == 0) return;
  if (blend != 2 || r.outputs.size() != 2)
    throw ShaderError(ErrorCode::StageInterface, std::format("{}: dual source blending writes blendOutput(0) and blendOutput(1), and no other output",
                                  r.program));
  if (!typesEqual(r.outputs[0].type, r.outputs[1].type))
    throw ShaderError(ErrorCode::TypeMismatch, std::format("{}: the two blend sources have different types, {} and {}", r.program,
                                  typeName(r.outputs[0].type), typeName(r.outputs[1].type)));
}

/** Attributes and varyings in order of first use; outputs by index. */
void reflectStageIO(Reflection& r, const StageUsage& vs, const StageUsage& fs) {
  for (const AttributePtr& a : vs.attributes)
    r.attributes.push_back({a->name, a->type, static_cast<int>(r.attributes.size()), a->step});
  // a varying that the fragment stage never reads would only use up interstage slots
  for (const VaryingPtr& v : vs.varyingsWritten)
    if (std::ranges::find(fs.varyingsRead, v) != fs.varyingsRead.end())
      r.varyings.push_back({v->name, v->type, static_cast<int>(r.varyings.size()), v->interpolation});
  for (const OutputPtr& o : fs.outputs) r.outputs.push_back({o->index, o->type, o->blendSrc});
  std::ranges::sort(r.outputs, {}, [](const ReflectedOutput& o) { return std::pair(o.index, o.blendSrc.value_or(0)); });
  checkDualSource(r);
}

/** Packs single uniforms, orders and binds every resource; returns them in binding order. */
std::vector<ResourcePtr> bindProgramResources(AnalyzedProgram& out, const AnalyzeOptions& options,
                                              const StageUsage& vs, const StageUsage& fs) {
  std::vector<ResourcePtr> requested = options.retain;
  std::vector<UniformPtr> usedUniforms = vs.uniforms;
  for (const UniformPtr& u : fs.uniforms) pushUnique(usedUniforms, u);
  packUniforms(usedUniforms, out, requested);
  requested.insert(requested.end(), vs.resources.begin(), vs.resources.end());
  requested.insert(requested.end(), fs.resources.begin(), fs.resources.end());
  std::vector<ResourcePtr> all = orderedResources(requested);
  checkResourceNames(all);
  bindResources(all, out);
  return all;
}

/** Functions (callees first), constants (inputs first) and helpers of both stages. */
void collectDeclarations(AnalyzedProgram& out, const StageUsage& vs, const StageUsage& fs) {
  for (const FunctionPtr& f : vs.functions) pushUnique(out.functions, f);
  for (const FunctionPtr& f : fs.functions) pushUnique(out.functions, f);
  for (const ConstantPtr& c : vs.constants) pushUnique(out.constants, c);
  for (const ConstantPtr& c : fs.constants) pushUnique(out.constants, c);
  out.vertexFunctions = vs.functions;
  out.fragmentFunctions = fs.functions;
  for (const Helper& h : vs.glslHelpers) pushUnique(out.glslHelpers, h);
  for (const Helper& h : fs.glslHelpers) pushUnique(out.glslHelpers, h);
  for (const Helper& h : vs.wgslHelpers) pushUnique(out.wgslHelpers, h);
  for (const Helper& h : fs.wgslHelpers) pushUnique(out.wgslHelpers, h);
  out.workgroups = vs.workgroups;
}

/** Every struct the program declares, nested structs first. */
void collectProgramStructs(AnalyzedProgram& out, const std::vector<ResourcePtr>& all,
                           const StageUsage& vs, const StageUsage& fs) {
  std::vector<TypePtr> types;
  for (const ResourcePtr& res : all) {
    if (res->kind == ResourceKind::UniformBlock)
      types.push_back(static_cast<const UniformBlockResource&>(*res).structType);
    else if (res->kind == ResourceKind::Table)
      types.push_back(static_cast<const TableResource&>(*res).structType);
    // a retained buffer is declared too, so its element struct must be
    else if (res->kind == ResourceKind::StorageBuffer)
      types.push_back(static_cast<const StorageBufferResource&>(*res).elementType);
  }
  for (const FunctionPtr& f : out.functions) {
    types.push_back(f->returnType);
    for (const LocalPtr& p : f->params) types.push_back(p->type);
  }
  types.insert(types.end(), vs.compositeTypes.begin(), vs.compositeTypes.end());
  types.insert(types.end(), fs.compositeTypes.begin(), fs.compositeTypes.end());
  for (const WorkgroupPtr& w : out.workgroups) types.push_back(w->type);
  collectStructs(types, out.structs);
}

/** Builtins, depth, clip distances and the stage kind. */
void reflectStages(AnalyzedProgram& out, const ProgramAST& ast, const StageUsage& vs,
                   const StageUsage& fs) {
  Reflection& r = out.reflection;
  r.usesFragDepth = fs.fragDepth;
  r.clipDistances = vs.clipDistances;
  r.compute = ast.compute;
  r.workgroupSize = ast.workgroupSize;
  r.subgroupSize = ast.subgroupSize;
  // subgroup-size-control needs subgroups on the device
  if (ast.subgroupSize) {
    pushUnique(r.deviceFeatures, DeviceFeature::Subgroups);
    pushUnique(r.deviceFeatures, DeviceFeature::SubgroupSizeControl);
  }
  if (ast.compute) {
    r.computeBuiltins = vs.builtins;
    out.computeFunctions = vs.functions;
    out.vertexFunctions.clear();
    r.features.push_back({Feature::ComputeProgram, "a compute program"});
  } else {
    r.vertexBuiltins = vs.builtins;
  }
  r.fragmentBuiltins = fs.builtins;
  for (const DeviceFeature f : vs.deviceFeatures) pushUnique(r.deviceFeatures, f);
  for (const DeviceFeature f : fs.deviceFeatures) pushUnique(r.deviceFeatures, f);
  for (const LanguageFeature f : vs.languageFeatures) pushUnique(r.languageFeatures, f);
}

/**
 * Features that a bound resource needs, whether a stage uses it or the
 * caller retains it (used resources are already in first use order): WebGPU validates them when it creates the bind group
 * layout, and the WGSL backend declares every bound resource.
 */
void recordResourceFeatures(AnalyzedProgram& out) {
  Reflection& r = out.reflection;
  for (const BoundResource& b : out.resources) {
    const Resource& res = *b.decl;
    if (res.kind == ResourceKind::Texture && static_cast<const TextureResource&>(res).float32Filterable)
      pushUnique(r.deviceFeatures, DeviceFeature::Float32Filterable);
    if (res.kind == ResourceKind::StorageBuffer) {
      pushUnique(r.features, FeatureUse{Feature::StorageBuffer, std::format("storage buffer {}", res.name)});
      // the declaration names the element type even when only length() or nothing reads it
      if (containsF16(static_cast<const StorageBufferResource&>(res).elementType)) {
        pushUnique(r.features, FeatureUse{Feature::F16, "f16"});
        pushUnique(r.deviceFeatures, DeviceFeature::ShaderF16);
      }
    }
    if (res.kind != ResourceKind::StorageTexture) continue;
    const auto& t = static_cast<const StorageTextureResource&>(res);
    pushUnique(r.features, FeatureUse{Feature::StorageTexture, std::format("storage texture {}", t.name)});
    for (const DeviceFeature f : storageFormatFeatures(t.format, t.access).value_or(std::vector<DeviceFeature>{}))
      pushUnique(r.deviceFeatures, f);
    if (t.access != StorageAccess::Write)
      pushUnique(r.languageFeatures, LanguageFeature::ReadonlyAndReadwriteStorageTextures);
    // the tier 1 formats are those whose write access needs texture-formats-tier1
    const auto write = storageFormatFeatures(t.format, StorageAccess::Write);
    if (write && std::ranges::find(*write, DeviceFeature::TextureFormatsTier1) != write->end())
      pushUnique(r.languageFeatures, LanguageFeature::TextureFormatsTier1);
  }
}

/** Records the features that not every backend has, first use first. */
void recordFeatures(AnalyzedProgram& out, const StageUsage& vs, const StageUsage& fs) {
  Reflection& r = out.reflection;
  std::vector<TypePtr> types = vs.compositeTypes;
  types.insert(types.end(), fs.compositeTypes.begin(), fs.compositeTypes.end());
  for (const TypePtr& st : out.structs)
    for (const StructField& f : st->fields) types.push_back(f.type);
  for (const FunctionPtr& f : out.functions) {
    types.push_back(f->returnType);
    for (const LocalPtr& p : f->params) types.push_back(p->type);
  }
  for (const WorkgroupPtr& w : out.workgroups) types.push_back(w->type);
  for (const TypePtr& t : types) {
    if (t->kind == TypeKind::Array && t->element->kind == TypeKind::Array)
      pushUnique(r.features, FeatureUse{Feature::ArrayOfArrays, std::format("array of arrays {}", typeName(t))});
    // a declared struct with an f16 field needs f16 even when no expression reads it
    if (containsF16(t)) {
      pushUnique(r.features, FeatureUse{Feature::F16, "f16"});
      pushUnique(r.deviceFeatures, DeviceFeature::ShaderF16);
    }
  }
  for (const ReflectedVarying& v : r.varyings)
    if (v.interpolation == Interpolation::Linear)
      pushUnique(r.features, FeatureUse{Feature::LinearInterpolation,
                                        std::format("linear interpolation of varying {}", v.name)});
  for (const FeatureUse& u : vs.features) pushUnique(r.features, u);
  for (const FeatureUse& u : fs.features) pushUnique(r.features, u);
  for (const ReflectedVarying& v : r.varyings)
    if (v.interpolation == Interpolation::Flat)
      pushUnique(r.features, FeatureUse{Feature::FlatFirstVertex, std::format("flat varying {}", v.name)});
}

} // namespace

std::optional<AnalyzedProgram> detail::analyzeCollecting(ProgramAST program, const AnalyzeOptions& options,
                                                         Errors& errors) {
  // these errors are about the whole program, not the C++ line last built
  detail::clearUserLocation();
  auto owned = std::make_shared<const ProgramAST>(std::move(program));
  const ProgramAST& ast = *owned;
  // a compute program has one stage; it goes through the vertex slot of the
  // two stage path, with the vertex only rules switched off by its stage name
  const std::size_t before = errors.size();
  UsageCollector vs(ast.compute ? StageName::Compute : StageName::Vertex, errors);
  vs.block(ast.compute ? ast.computeBody : ast.vertex, nullptr);
  UsageCollector fs(StageName::Fragment, errors);
  fs.block(ast.fragment, nullptr);
  checkStageInterface(vs.usage, fs.usage, errors);
  detail::checkLocalScopes(ast, errors);
  if (!ast.compute) checkPositionWritten(ast, errors);
  detail::checkUniformity(ast, errors);
  // the passes below assume a valid program
  if (errors.size() != before) return std::nullopt;

  AnalyzedProgram out;
  out.ast = std::move(owned);
  out.reflection.program = ast.name;
  out.reflection.variantKey = ast.variantKey;
  reflectStageIO(out.reflection, vs.usage, fs.usage);
  const std::vector<ResourcePtr> all = bindProgramResources(out, options, vs.usage, fs.usage);
  collectDeclarations(out, vs.usage, fs.usage);
  collectProgramStructs(out, all, vs.usage, fs.usage);
  reflectStages(out, ast, vs.usage, fs.usage);
  recordFeatures(out, vs.usage, fs.usage);
  recordResourceFeatures(out);
  checkSharedStructs(out);
  checkModuleNames(out);
  return out;
}

AnalyzedProgram analyzeProgram(ProgramAST program, const AnalyzeOptions& options) {
  detail::Errors errors;
  std::optional<AnalyzedProgram> out = detail::analyzeCollecting(std::move(program), options, errors);
  if (!errors.empty()) throw errors.front();
  return std::move(*out);
}

} // namespace moga
