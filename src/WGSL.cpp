#include "BuiltinTable.hpp"
#include "Steps.hpp"
#include "Lower.hpp"
#include "TextEmitter.hpp"

#include <array>
#include "Walk.hpp"
#include "moga/Compile.hpp"

#include <cstdint>
#include <format>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace moga {

using namespace detail;

namespace {

constexpr std::string_view kVertexOut = "vout";
constexpr std::string_view kVertexIn = "vin";
constexpr std::string_view kFragmentOut = "fout";
constexpr std::string_view kFragmentIn = "fin";

char scalarSuffix(ScalarKind k) {
  switch (k) {
  case ScalarKind::I32:
    return 'i';
  case ScalarKind::U32:
    return 'u';
  case ScalarKind::F16:
    return 'h';
  default:
    return 'f';
  }
}

std::string_view callName(BuiltinFunction fn) { return builtinSpec(fn).wgsl; }

bool isBitcast(BuiltinFunction fn) { return builtinSpec(fn).bitcast; }

std::string_view textureBase(TextureDimension d) {
  switch (d) {
  case TextureDimension::D2:
    return "texture_2d";
  case TextureDimension::D3:
    return "texture_3d";
  case TextureDimension::D2Array:
    return "texture_2d_array";
  case TextureDimension::Cube:
    return "texture_cube";
  }
  return "?";
}

std::string_view sampleScalar(TextureSampleType s) {
  if (s == TextureSampleType::Uint) return "u32";
  if (s == TextureSampleType::Sint) return "i32";
  return "f32";
}

std::string textureType(const TextureResource& t) {
  if (t.sampleType == TextureSampleType::Depth) {
    switch (t.dimension) {
    case TextureDimension::D2:
      return "texture_depth_2d";
    case TextureDimension::D2Array:
      return "texture_depth_2d_array";
    case TextureDimension::Cube:
      return "texture_depth_cube";
    case TextureDimension::D3:
      break;
    }
    throw ShaderError(ErrorCode::InvalidDeclaration, std::format("{}: there are no 3D depth textures", t.name));
  }
  return std::format("{}<{}>", textureBase(t.dimension), sampleScalar(t.sampleType));
}

constexpr std::string_view kComputeIn = "cin";

std::string_view accessName(StorageAccess a) {
  switch (a) {
  case StorageAccess::Read:
    return "read";
  case StorageAccess::Write:
    return "write";
  case StorageAccess::ReadWrite:
    return "read_write";
  }
  return "?";
}

std::string_view storageTextureBase(TextureDimension d) {
  switch (d) {
  case TextureDimension::D3:
    return "texture_storage_3d";
  case TextureDimension::D2Array:
    return "texture_storage_2d_array";
  default:
    return "texture_storage_2d";
  }
}

/** WGSL attribute name of a builtin input, for example "global_invocation_id". */
std::string_view builtinAttribute(BuiltinInput b) {
  switch (b) {
  case BuiltinInput::GlobalId:
    return "global_invocation_id";
  case BuiltinInput::LocalId:
    return "local_invocation_id";
  case BuiltinInput::WorkgroupId:
    return "workgroup_id";
  case BuiltinInput::NumWorkgroups:
    return "num_workgroups";
  case BuiltinInput::LocalIndex:
    return "local_invocation_index";
  case BuiltinInput::SubgroupSize:
    return "subgroup_size";
  case BuiltinInput::SubgroupInvocationId:
    return "subgroup_invocation_id";
  case BuiltinInput::PrimitiveIndex:
    return "primitive_index";
  default:
    return builtinInputName(b);
  }
}

std::string builtinInput(BuiltinInput name, StageName stage) {
  if (isComputeBuiltin(name) || (isSubgroupBuiltin(name) && stage == StageName::Compute))
    return std::format("{}.{}", kComputeIn, builtinInputName(name));
  switch (name) {
  case BuiltinInput::VertexIndex:
  case BuiltinInput::InstanceIndex:
    return std::format("{}.{}", kVertexIn, builtinInputName(name));
  case BuiltinInput::DrawIndex:
  case BuiltinInput::LocalInstanceIndex:
    throw ShaderError(ErrorCode::MissingFeature, std::format(
        "{} is GLSL-only; WebGPU variants use instanceIndex with firstInstance",
        builtinInputName(name)));
  default:
    return std::format("{}.{}", kFragmentIn, builtinInputName(name));
  }
}

/** WGSL shift amounts must be unsigned. */
std::string unsignedShiftAmount(const std::string& amount, const TypePtr& t) {
  if ((t->kind != TypeKind::Scalar && t->kind != TypeKind::Vector) || t->scalar == ScalarKind::U32)
    return amount;
  return t->kind == TypeKind::Vector ? std::format("vec{}u({})", t->size, amount)
                                     : std::format("u32({})", amount);
}

std::string_view interp(Interpolation i) {
  if (i == Interpolation::Flat) return " @interpolate(flat, first)";
  if (i == Interpolation::FlatEither) return " @interpolate(flat, either)";
  if (i == Interpolation::Linear) return " @interpolate(linear)";
  return "";
}

/** Adds the names of a struct and of the structs nested in it. */
void addStructNames(const TypeInfo& s, std::set<std::string>& out) {
  out.insert(s.name);
  for (const StructField& f : s.fields) {
    const TypeInfo* t = f.type.get();
    while (t->kind == TypeKind::Array) t = t->element.get();
    if (t->kind == TypeKind::Struct) addStructNames(*t, out);
  }
}

/** Helper names carry the matrix type, so f32 and f16 helpers of one size can both exist. */
std::string inverseName(const TypePtr& m) { return std::format("moga_inverse_{}", wgslType(m)); }
std::string mat3FromMat4Name(const TypePtr& m) {
  return std::format("moga_{}_from_{}", wgslType(m), wgslType(squareMatrix(4, m->scalar)));
}

std::string literal(const std::variant<double, bool>& value, const TypePtr& t) {
  if (const bool* b = std::get_if<bool>(&value)) return *b ? "true" : "false";
  const double v = std::get<double>(value);
  if (t->kind != TypeKind::Scalar) throw ShaderError(ErrorCode::Internal, "Literal must be scalar");
  std::string s;
  if (t->scalar == ScalarKind::F32) s = formatFloat(v) + "f";
  else if (t->scalar == ScalarKind::F16) s = formatFloat(v) + "h";
  else if (t->scalar == ScalarKind::U32)
    s = std::format("{}u", static_cast<std::uint32_t>(static_cast<std::int64_t>(v)));
  else s = std::format("{}i", static_cast<std::int32_t>(static_cast<std::int64_t>(v)));
  // 2147483648i is out of range before the minus applies; the abstract literal converts
  if (t->scalar == ScalarKind::I32 && v == static_cast<double>(std::numeric_limits<std::int32_t>::min()))
    return "i32(-2147483648)";
  return v < 0 ? "(" + s + ")" : s;
}

class WGSLEmitter {
public:
  explicit WGSLEmitter(const AnalyzedProgram& program) : program_(program) {}

  TextEmitter out;

  void emit() {
    const Reflection& r = program_.reflection;
    std::string required;
    for (const LanguageFeature f : r.languageFeatures)
      required += std::format("{}{}", required.empty() ? "" : ", ", languageFeatureName(f));
    if (!required.empty()) out.line(std::format("requires {};", required));
    // the only source of enable directives, so each one is a reported device feature
    for (const DeviceFeature f : r.deviceFeatures)
      if (const auto extension = wgslExtension(f)) out.line(std::format("enable {};", *extension));
    out.line();

    // structs are compared by name: equal definitions may be different objects
    std::set<std::string> uniformStructs;
    for (const ReflectedUniformBlock& ub : r.uniformBlocks)
      addStructNames(*ub.layout->structType, uniformStructs);
    for (const TypePtr& s : program_.structs) structDecl(s, uniformStructs.contains(s->name));

    bindings();
    for (const WorkgroupPtr& w : program_.workgroups)
      out.line(std::format("var<workgroup> wg_{}: {};", sanitize(w->name), wgslType(w->type)));
    out.line();
    if (r.compute) computeInput();
    else ioStructs();
    for (const ConstantPtr& c : program_.constants) constantDecl(*c);
    for (const Helper& h : program_.wgslHelpers) helperDecl(h.kind, h.type);
    for (const FunctionPtr& f : program_.functions) func(*f);
    if (r.compute) {
      computeEntry();
      return;
    }
    entry(StageName::Vertex);
    entry(StageName::Fragment);
  }

  void computeInput() {
    const Reflection& r = program_.reflection;
    // WGSL structs need a member; cs_main then takes no parameter
    if (r.computeBuiltins.empty()) return;
    out.line("struct CSIn {");
    out.indent();
    for (const BuiltinInput b : r.computeBuiltins)
      out.line(std::format("@builtin({}) {}: {},", builtinAttribute(b), builtinInputName(b),
                           b == BuiltinInput::LocalIndex || isSubgroupBuiltin(b) ? "u32" : "vec3u"));
    out.dedent();
    out.line("}");
    out.line();
  }

  void computeEntry() {
    const Reflection& r = program_.reflection;
    stage_ = StageName::Compute;
    const auto& [x, y, z] = r.workgroupSize;
    const std::string subgroup = r.subgroupSize ? std::format(" @subgroup_size({})", *r.subgroupSize) : "";
    out.line(std::format("@compute @workgroup_size({}, {}, {}){} fn cs_main({}) {{", x, y, z, subgroup,
                         r.computeBuiltins.empty() ? "" : std::format("{}: CSIn", kComputeIn)));
    out.indent();
    block(program_.ast->computeBody);
    out.dedent();
    out.line("}");
  }

private:
  void bindings() {
    for (const BoundResource& b : program_.resources) {
      const std::string g =
          std::format("@group({}) @binding({})", groupOf(b.reflected), bindingOf(b.reflected));
      const std::string name = sanitize(b.decl->name);
      switch (b.decl->kind) {
      case ResourceKind::UniformBlock: {
        const auto& ub = static_cast<const UniformBlockResource&>(*b.decl);
        out.line(std::format("{} var<uniform> ub_{}: {};", g, name, structName(ub.structType->name)));
        break;
      }
      case ResourceKind::Table: {
        const auto& tb = static_cast<const TableResource&>(*b.decl);
        out.line(std::format("{} var<storage, read> tb_{}: array<{}>;", g, name,
                             structName(tb.structType->name)));
        break;
      }
      case ResourceKind::Texture: {
        const auto& t = static_cast<const TextureResource&>(*b.decl);
        out.line(std::format("{} var t_{}: {};", g, name, textureType(t)));
        const auto& rt = std::get<ReflectedTexture>(b.reflected);
        if (rt.samplerBinding >= 0)
          out.line(std::format("@group({}) @binding({}) var s_{}: {};", rt.group,
                               rt.samplerBinding, name,
                               rt.sampler == SamplerBinding::Comparison ? "sampler_comparison" : "sampler"));
        break;
      }
      case ResourceKind::StorageBuffer: {
        const auto& sb = static_cast<const StorageBufferResource&>(*b.decl);
        out.line(std::format("{} var<storage, {}> sb_{}: array<{}>;", g, accessName(sb.access),
                             name, wgslType(sb.elementType)));
        break;
      }
      case ResourceKind::StorageTexture: {
        const auto& st = static_cast<const StorageTextureResource&>(*b.decl);
        out.line(std::format("{} var st_{}: {}<{}, {}>;", g, name, storageTextureBase(st.dimension),
                             texelFormatName(st.format), accessName(st.access)));
        break;
      }
      }
    }
  }

  void structDecl(const TypePtr& s, bool explicitLayout) {
    out.line(std::format("struct {} {{", structName(s->name)));
    out.indent();
    if (explicitLayout) {
      // Pin offsets to the shared std140 compatible layout.
      const StructLayout& l = *structLayout(s);
      for (std::size_t i = 0; i < l.fields.size(); ++i) {
        const FieldLayout& f = l.fields[i];
        const std::size_t next = i + 1 < l.fields.size() ? l.fields[i + 1].offset : l.size;
        out.line(std::format("@align({}) @size({}) {}: {},", f.align, next - f.offset,
                             sanitize(f.name), wgslType(f.type)));
      }
    } else {
      for (const StructField& f : s->fields)
        out.line(std::format("{}: {},", sanitize(f.name), wgslType(f.type)));
    }
    out.dedent();
    out.line("}");
  }

  /**
   * WGSL rejects a constant expression that gives inf or NaN, so a bitcast
   * constant becomes a function that does the bitcast at run time.
   */
  void constantDecl(const ConstantDecl& c) {
    const std::string name = constantName(c.name);
    const auto* call = std::get_if<ex::Call>(&c.value->node);
    if (call == nullptr || !isBitcast(call->fn)) {
      // a const cannot call a function, so a constant that reads a runtime
      // constant becomes a function as well (constants come after their inputs)
      // a reused value inside the constant is printed once, as a helper const
      // (or a let, inside the function of a runtime constant)
      if (readsRuntimeConstant(*c.value) && sharedSubexpressions(c.value).empty()) {
        out.line(std::format("fn {}() -> {} {{ return {}; }}", name, wgslType(c.type), expr(*c.value)));
        runtimeConstants_.insert(&c);
        return;
      }
      if (readsRuntimeConstant(*c.value)) {
        out.line(std::format("fn {}() -> {} {{", name, wgslType(c.type)));
        out.indent();
        const auto added = declareShared(c.value, "let");
        out.line(std::format("return {};", expr(*c.value)));
        out.dedent();
        out.line("}");
        forget(added);
        runtimeConstants_.insert(&c);
        return;
      }
      inConstant_ = true;
      const auto added = declareShared(c.value, "const");
      const std::string value = expr(*c.value);
      inConstant_ = false;
      out.line(std::format("const {}: {} = {};", name, wgslType(c.type), value));
      forget(added);
      return;
    }
    const ExprPtr& bits = call->args[0];
    if (!std::holds_alternative<ex::Lit>(bits->node))
      throw ShaderError(ErrorCode::InvalidConstant, std::format("Constant {}: a bitcast constant must take a literal", c.name));
    out.line(std::format("fn {}() -> {} {{", name, wgslType(c.type)));
    out.indent();
    out.line(std::format("var bits: {} = {};", wgslType(bits->type), expr(*bits)));
    out.line(std::format("return bitcast<{}>(bits);", wgslType(c.type)));
    out.dedent();
    out.line("}");
    runtimeConstants_.insert(&c);
  }

  /** Prints each reused part of a constant once, as `keyword moga_shared_N: T = ...;`. */
  std::vector<const Expr*> declareShared(const ExprPtr& value, std::string_view keyword) {
    std::vector<const Expr*> added;
    for (const Expr* e : sharedSubexpressions(value)) {
      if (shared_.contains(e)) continue;
      const std::string local = std::format("moga_shared_{}", sharedCount_++);
      out.line(std::format("{} {}: {} = {};", keyword, local, wgslType(e->type), exprBody(*e)));
      shared_.emplace(e, local);
      added.push_back(e);
    }
    return added;
  }

  void forget(const std::vector<const Expr*>& added) {
    for (const Expr* e : added) shared_.erase(e);
  }

  /** Whether the value reads a constant printed as a function; each shared node is checked once. */
  bool readsRuntimeConstant(const Expr& e) const {
    std::unordered_set<const Expr*> seen;
    return readsRuntimeConstant(e, seen);
  }

  bool readsRuntimeConstant(const Expr& e, std::unordered_set<const Expr*>& seen) const {
    if (const auto* k = std::get_if<ex::Constant>(&e.node)) return runtimeConstants_.contains(k->c.get());
    if (!seen.insert(&e).second) return false;
    bool found = false;
    forEachOperand(e, [&](const Expr& operand) { found = found || readsRuntimeConstant(operand, seen); });
    return found;
  }

  void ioStructs() {
    const Reflection& r = program_.reflection;
    vertexInput();
    vertexOutput();
    fragmentInput();
    hasFragmentOutput_ = !r.outputs.empty() || r.usesFragDepth;
    if (hasFragmentOutput_) fragmentOutput();
    out.line();
  }

  void vertexInput() {
    const Reflection& r = program_.reflection;
    // WGSL structs need a member; vs_main then takes no parameter
    if (r.attributes.empty() && !r.usesVertexBuiltin(BuiltinInput::VertexIndex) &&
        !r.usesVertexBuiltin(BuiltinInput::InstanceIndex))
      return;
    out.line("struct VSIn {");
    out.indent();
    for (const ReflectedAttribute& a : r.attributes)
      out.line(std::format("@location({}) a_{}: {},", a.location, sanitize(a.name), wgslType(a.type)));
    if (r.usesVertexBuiltin(BuiltinInput::VertexIndex))
      out.line("@builtin(vertex_index) vertexIndex: u32,");
    if (r.usesVertexBuiltin(BuiltinInput::InstanceIndex))
      out.line("@builtin(instance_index) instanceIndex: u32,");
    out.dedent();
    out.line("}");
  }

  void vertexOutput() {
    const Reflection& r = program_.reflection;
    out.line("struct VSOut {");
    out.indent();
    out.line("@builtin(position) position: vec4f,");
    for (const ReflectedVarying& v : r.varyings)
      out.line(std::format("@location({}){} v_{}: {},", v.location, interp(v.interpolation),
                           sanitize(v.name), wgslType(v.type)));
    if (r.clipDistances > 0)
      out.line(std::format("@builtin(clip_distances) clipDistances: array<f32, {}>,", r.clipDistances));
    out.dedent();
    out.line("}");
  }

  void fragmentInput() {
    const Reflection& r = program_.reflection;
    // WGSL structs need a member; fs_main then takes no parameter
    if (r.varyings.empty() && r.fragmentBuiltins.empty()) return;
    out.line("struct FSIn {");
    out.indent();
    if (r.usesFragmentBuiltin(BuiltinInput::FragCoord))
      out.line("@builtin(position) fragCoord: vec4f,");
    if (r.usesFragmentBuiltin(BuiltinInput::FrontFacing))
      out.line("@builtin(front_facing) frontFacing: bool,");
    for (const BuiltinInput b :
         {BuiltinInput::SubgroupSize, BuiltinInput::SubgroupInvocationId, BuiltinInput::PrimitiveIndex})
      if (r.usesFragmentBuiltin(b))
        out.line(std::format("@builtin({}) {}: u32,", builtinAttribute(b), builtinInputName(b)));
    for (const ReflectedVarying& v : r.varyings)
      out.line(std::format("@location({}){} v_{}: {},", v.location, interp(v.interpolation),
                           sanitize(v.name), wgslType(v.type)));
    out.dedent();
    out.line("}");
  }

  void fragmentOutput() {
    const Reflection& r = program_.reflection;
    out.line("struct FSOut {");
    out.indent();
    for (const ReflectedOutput& o : r.outputs)
      out.line(std::format("@location({}){} {}: {},", o.index,
                           o.blendSrc ? std::format(" @blend_src({})", *o.blendSrc) : "",
                           outputName(o.index, o.blendSrc), wgslType(o.type)));
    if (r.usesFragDepth) out.line("@builtin(frag_depth) fragDepth: f32,");
    out.dedent();
    out.line("}");
  }

  void entry(StageName stage) {
    const Reflection& r = program_.reflection;
    stage_ = stage;
    if (stage == StageName::Vertex) {
      const bool hasInput = !r.attributes.empty() || !r.vertexBuiltins.empty();
      out.line(std::format("@vertex fn vs_main({}) -> VSOut {{",
                           hasInput ? std::format("{}: VSIn", kVertexIn) : ""));
      out.indent();
      out.line(std::format("var {}: VSOut;", kVertexOut));
      block(program_.ast->vertex);
      out.line(std::format("return {};", kVertexOut));
    } else {
      const bool hasInput = !r.varyings.empty() || !r.fragmentBuiltins.empty();
      out.line(std::format("@fragment fn fs_main({}){} {{",
                           hasInput ? std::format("{}: FSIn", kFragmentIn) : "",
                           hasFragmentOutput_ ? " -> FSOut" : ""));
      out.indent();
      if (hasFragmentOutput_) out.line(std::format("var {}: FSOut;", kFragmentOut));
      block(program_.ast->fragment);
      if (hasFragmentOutput_) out.line(std::format("return {};", kFragmentOut));
    }
    out.dedent();
    out.line("}");
  }

  void func(const FunctionDecl& f) {
    std::vector<std::string> params;
    for (const LocalPtr& p : f.params)
      params.push_back(std::format("{}: {}", localName(*p), wgslType(p->type)));
    const std::string ret =
        f.returnType->kind == TypeKind::Void ? "" : " -> " + wgslType(f.returnType);
    out.line(std::format("fn {}({}){} {{", functionName(f), join(params, ", "), ret));
    out.indent();
    fn_ = &f;
    block(f.body);
    fn_ = nullptr;
    out.dedent();
    out.line("}");
  }

  void block(const Block& stmts) {
    for (const StmtPtr& s : stmts) stmt(*s);
  }

  void nested(std::string_view header, const Block& body, const std::optional<DebugInfo>& debug) {
    out.line(header, debug);
    out.indent();
    block(body);
    out.dedent();
    out.line("}");
  }

  std::string stageReturn() const {
    if (stage_ == StageName::Compute) return "return;";
    if (stage_ == StageName::Vertex) return std::format("return {};", kVertexOut);
    return hasFragmentOutput_ ? std::format("return {};", kFragmentOut) : "return;";
  }

  /**
   * Prints the statement after the values that it reads more than once,
   * each as a local, so a reused value is computed and printed once.
   */
  void stmt(const Stmt& s) {
    std::vector<const Expr*> added;
    for (const Expr* e : sharedSubexpressions(s)) {
      // an enclosing statement may have printed it already, and its local is still in scope
      if (shared_.contains(e)) continue;
      const std::string name = std::format("moga_shared_{}", sharedCount_++);
      out.line(std::format("let {}: {} = {};", name, wgslType(e->type), exprBody(*e)), s.debug);
      shared_.emplace(e, name);
      added.push_back(e);
    }
    stmtBody(s);
    for (const Expr* e : added) shared_.erase(e);
  }

  std::string expr(const Expr& e) {
    if (const auto it = shared_.find(&e); it != shared_.end()) return it->second;
    return exprBody(e);
  }

  void stmtBody(const Stmt& s) {
    const auto& d = s.debug;
    std::visit(
        Overloaded{
            [&](const st::Decl& x) {
              const std::string name = localName(*x.v);
              const std::string type = wgslType(x.v->type);
              if (!x.init) return out.line(std::format("var {}: {};", name, type), d);
              // a let array cannot be indexed with a runtime index
              const bool var = x.v->mutableVar || x.v->type->kind == TypeKind::Array;
              out.line(std::format("{} {}: {} = {};", var ? "var" : "let", name, type,
                                   expr(*x.init)),
                       d);
            },
            [&](const st::Assign& x) { assignment(x, d); },
            [&](const st::If& x) { ifStatement(x, d); },
            [&](const st::For& x) {
              const std::string i = localName(*x.v);
              nested(std::format("for (var {}: {} = {}; {} {} {}; {} = {} + {}) {{", i,
                                 wgslType(x.v->type), expr(*x.start), i, opText(x.condition),
                                 expr(*x.end), i, i, expr(*x.step)),
                     x.body, d);
            },
            [&](const st::While& x) {
              nested(std::format("while ({}) {{", expr(*x.cond)), x.body, d);
            },
            [&](const st::Switch& x) { switchStatement(x, d); },
            [&](const st::Loop& x) { nested("loop {", x.body, d); },
            [&](const st::Break&) { out.line("break;", d); },
            [&](const st::Continue&) { out.line("continue;", d); },
            [&](const st::Discard&) { out.line("discard;", d); },
            [&](const st::Return& x) {
              if (fn_ == nullptr) return out.line(stageReturn(), d);
              out.line(x.value ? std::format("return {};", expr(*x.value)) : "return;", d);
            },
            [&](const st::SetVarying& x) {
              // a varying the fragment stage never reads is not declared; a call in its value still runs
              if (program_.reflection.hasVarying(x.v->name))
                out.line(std::format("{}.v_{} = {};", kVertexOut, sanitize(x.v->name), expr(*x.value)), d);
              else if (hasSideEffects(*x.value))
                out.line(std::format("_ = {};", expr(*x.value)), d);
            },
            [&](const st::SetOutput& x) {
              out.line(std::format("{}.{} = {};", kFragmentOut, outputName(x.o->index, x.o->blendSrc),
                                   expr(*x.value)), d);
            },
            [&](const st::SetPosition& x) {
              out.line(std::format("{}.position = {};", kVertexOut, expr(*x.value)), d);
            },
            [&](const st::SetFragDepth& x) {
              out.line(std::format("{}.fragDepth = {};", kFragmentOut, expr(*x.value)), d);
            },
            [&](const st::SetClipDistance& x) {
              out.line(std::format("{}.clipDistances[{}] = {};", kVertexOut, x.index, expr(*x.value)), d);
            },
            // WGSL forbids bare value expressions; only void calls reach here.
            [&](const st::ExprStmt& x) { out.line(expr(*x.e) + ";", d); },
            [&](const st::Barrier& x) {
              static constexpr std::array<std::string_view, 3> names{
                  "workgroupBarrier", "storageBarrier", "textureBarrier"};
              out.line(std::format("{}();", names[static_cast<int>(x.kind)]), d);
            },
            [&](const st::AtomicStore& x) {
              out.line(std::format("atomicStore(&{}, {});", expr(*x.target), expr(*x.value)), d);
            },
            [&](const st::TextureStore& x) {
              const std::string layer = x.layer ? ", " + expr(*x.layer) : "";
              out.line(std::format("textureStore(st_{}, {}{}, {});", sanitize(x.tex->name),
                                   expr(*x.coord), layer, expr(*x.value)),
                       d);
            },
        },
        s.node);
  }

  /**
   * WGSL writes one swizzle component at a time (a multi component swizzle
   * write needs the swizzle_assignment extension): the value goes in a
   * temporary and each component is assigned from it through a pointer to the
   * base. Swizzles of swizzles are merged when built, so the base is never a
   * swizzle.
   */
  void assignment(const st::Assign& x, const std::optional<DebugInfo>& debug) {
    const auto* s = std::get_if<ex::Swizzle>(&x.target->node);
    const std::string op = x.op ? std::string(opText(*x.op)) : "";
    // a compound shift takes a u32 amount
    const bool shift = x.op == BinaryOp::Shl || x.op == BinaryOp::Shr;
    const std::string value = shift ? unsignedShiftAmount(expr(*x.value), x.value->type) : expr(*x.value);
    if (s == nullptr || s->comps.size() == 1) {
      out.line(std::format("{} {}= {};", expr(*x.target), op, value), debug);
      return;
    }
    // the pointer evaluates the base (and any index in it) once
    out.line("{", debug);
    out.indent();
    // the target first, then the value: the left to right order of an assignment
    out.line(std::format("let moga_target = &{};", expr(*s->base)));
    out.line(std::format("let moga_swizzle = {};", value));
    for (std::size_t i = 0; i < s->comps.size(); ++i)
      out.line(std::format("(*moga_target).{} {}= moga_swizzle.{};", s->comps[i], op, "xyzw"[i]));
    out.dedent();
    out.line("}");
  }

  /** WGSL needs a default clause, so an empty one is added when there is none. */
  void switchStatement(const st::Switch& s, const std::optional<DebugInfo>& debug) {
    out.line(std::format("switch ({}) {{", expr(*s.selector)), debug);
    out.indent();
    for (const st::SwitchCase& c : s.cases) {
      out.line(std::format("case {}: {{", join(exprs(c.values), ", ")));
      out.indent();
      block(c.body);
      out.dedent();
      out.line("}");
    }
    out.line("default: {");
    out.indent();
    if (s.defaultBody) block(*s.defaultBody);
    out.dedent();
    out.line("}");
    out.dedent();
    out.line("}");
  }

  /** WGSL has no matrix inverse, so each used size gets a helper function. */
  void helperDecl(HelperKind kind, const TypePtr& type) {
    if (kind == HelperKind::FloatMod) {
      const std::string t = wgslType(type);
      out.line(std::format("fn moga_mod_{}(a: {}, b: {}) -> {} {{ return a - b * floor(a / b); }}", t, t, t, t));
      return;
    }
    if (kind == HelperKind::Mat3FromMat4) {
      const std::string t = wgslType(type);
      const std::string from = wgslType(squareMatrix(4, type->scalar));
      out.line(std::format("fn {}(m: {}) -> {} {{ return {}(m[0].xyz, m[1].xyz, m[2].xyz); }}",
                           mat3FromMat4Name(type), from, t, t));
      return;
    }
    if (kind != HelperKind::Inverse) return;
    const int n = type->cols;
    const std::string t = wgslType(type);
    out.line(std::format("fn {}(m: {}) -> {} {{", inverseName(type), t, t));
    out.indent();
    if (n == 2) {
      out.line("let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];");
      out.line(std::format("return {}(m[1][1], -m[0][1], -m[1][0], m[0][0]) * (1.0 / det);", t));
    } else if (n == 3) {
      // the rows of the inverse are the cross products of the columns
      out.line("let a = m[0];");
      out.line("let b = m[1];");
      out.line("let c = m[2];");
      out.line("let det = dot(a, cross(b, c));");
      out.line(std::format("return transpose({}(cross(b, c), cross(c, a), cross(a, b))) * (1.0 / det);", t));
    } else {
      // cofactor expansion by 2x2 minors
      for (int c = 0; c < 4; ++c)
        out.line(std::format("let a{0}0 = m[{0}][0]; let a{0}1 = m[{0}][1]; "
                             "let a{0}2 = m[{0}][2]; let a{0}3 = m[{0}][3];",
                             c));
      for (const char* line : {
               "let b00 = a00 * a11 - a01 * a10;",
               "let b01 = a00 * a12 - a02 * a10;",
               "let b02 = a00 * a13 - a03 * a10;",
               "let b03 = a01 * a12 - a02 * a11;",
               "let b04 = a01 * a13 - a03 * a11;",
               "let b05 = a02 * a13 - a03 * a12;",
               "let b06 = a20 * a31 - a21 * a30;",
               "let b07 = a20 * a32 - a22 * a30;",
               "let b08 = a20 * a33 - a23 * a30;",
               "let b09 = a21 * a32 - a22 * a31;",
               "let b10 = a21 * a33 - a23 * a31;",
               "let b11 = a22 * a33 - a23 * a32;",
               "let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;",
               "  a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09,",
               "  a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03,",
               "  a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07,",
               "  a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01,",
               "  a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06,",
               "  a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00,",
               "  a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06,",
               "  a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * (1.0 / det);",
           }) {
        out.line(line);
        if (std::string_view(line).starts_with("let det")) out.line(std::format("return {}(", t));
      }
    }
    out.dedent();
    out.line("}");
  }

  void ifStatement(const st::If& s, const std::optional<DebugInfo>& debug) {
    out.line(std::format("if ({}) {{", expr(*s.cond)), debug);
    out.indent();
    block(s.body);
    out.dedent();
    if (s.elseBody) {
      out.line("} else {");
      out.indent();
      block(*s.elseBody);
      out.dedent();
    }
    out.line("}");
  }

  std::vector<std::string> exprs(const std::vector<ExprPtr>& list) {
    std::vector<std::string> parts;
    for (const ExprPtr& e : list) parts.push_back(expr(*e));
    return parts;
  }

  std::string exprBody(const Expr& e) {
    return std::visit(
        Overloaded{
            [&](const ex::Lit& x) { return literal(x.value, e.type); },
            [&](const ex::Local& x) { return localName(*x.v); },
            [&](const ex::Constant& x) {
              const std::string name = constantName(x.c->name);
              return runtimeConstants_.contains(x.c.get()) ? name + "()" : name;
            },
            [&](const ex::Attribute& x) { return std::format("{}.a_{}", kVertexIn, sanitize(x.a->name)); },
            [&](const ex::Varying& x) { return std::format("{}.v_{}", kFragmentIn, sanitize(x.v->name)); },
            [&](const ex::Uniform& x) {
              return std::format("ub_{}.{}", sanitize(x.r->name), sanitize(x.field));
            },
            [&](const ex::LooseUniform& x) {
              const UniformBlockResource* block = program_.blockOf(x.u.get());
              if (block == nullptr)
                throw ShaderError(ErrorCode::Internal, 
                    std::format("WGSL: uniform {} is not part of the analyzed program", x.u->name));
              return std::format("ub_{}.{}", sanitize(block->name), sanitize(x.u->name));
            },
            [&](const ex::Builtin& x) { return builtinInput(x.name, stage_); },
            [&](const ex::Member& x) { return std::format("{}.{}", expr(*x.base), sanitize(x.field)); },
            [&](const ex::Swizzle& x) {
              const std::string base = expr(*x.base);
              if (x.base->type->kind == TypeKind::Scalar)
                return x.comps.size() == 1 ? base : std::format("{}({})", wgslType(e.type), base);
              return std::format("{}.{}", base, x.comps);
            },
            [&](const ex::Index& x) { return std::format("{}[{}]", expr(*x.base), expr(*x.index)); },
            [&](const ex::Unary& x) { return std::format("({}{})", opText(x.op), expr(*x.a)); },
            [&](const ex::Binary& x) {
              const std::string a = expr(*x.a);
              const std::string b = expr(*x.b);
              const bool shift = x.op == BinaryOp::Shl || x.op == BinaryOp::Shr;
              return std::format("({} {} {})", a, opText(x.op),
                                 shift ? unsignedShiftAmount(b, x.b->type) : b);
            },
            [&](const ex::Call& x) { return call(e, x); },
            [&](const ex::UCall& x) {
              return std::format("{}({})", functionName(*x.fn), join(exprs(x.args), ", "));
            },
            [&](const ex::Construct& x) { return construct(e, x.args); },
            [&](const ex::Select& x) {
              return std::format("select({}, {}, {})", expr(*x.f), expr(*x.t), expr(*x.cond));
            },
            [&](const ex::Sample& x) { return sample(x); },
            [&](const ex::Load& x) {
              const std::string layer = x.layer ? ", " + expr(*x.layer) : "";
              return std::format("textureLoad(t_{}, {}{}, {})", sanitize(x.tex->name),
                                 expr(*x.coord), layer, expr(*x.level));
            },
            [&](const ex::TexSize& x) {
              const std::string t = "t_" + sanitize(x.tex->name);
              const std::string dims = std::format("textureDimensions({}, {})", t, expr(*x.level));
              // WGSL reports array layers separately (GLSL textureSize includes them)
              if (x.tex->dimension == TextureDimension::D2Array)
                return std::format("vec3i(vec2i({}), i32(textureNumLayers({})))", dims, t);
              const int n = e.type->kind == TypeKind::Vector ? e.type->size : 2;
              return std::format("vec{}i({})", n, dims);
            },
            [&](const ex::TableRead& x) {
              return std::format("tb_{}[{}].{}", sanitize(x.table->name), expr(*x.index),
                                 sanitize(x.field));
            },
            [&](const ex::StorageElement& x) {
              return std::format("sb_{}[{}]", sanitize(x.buffer->name), expr(*x.index));
            },
            [&](const ex::StorageLength& x) {
              return std::format("arrayLength(&sb_{})", sanitize(x.buffer->name));
            },
            [&](const ex::Workgroup& x) { return "wg_" + sanitize(x.w->name); },
            [&](const ex::Atomic& x) {
              if (!x.value) return std::format("{}(&{})", atomicOpName(x.op), expr(*x.target));
              return std::format("{}(&{}, {})", atomicOpName(x.op), expr(*x.target), expr(*x.value));
            },
            [&](const ex::StorageLoad& x) {
              const std::string layer = x.layer ? ", " + expr(*x.layer) : "";
              return std::format("textureLoad(st_{}, {}{})", sanitize(x.tex->name), expr(*x.coord),
                                 layer);
            },
            [&](const ex::StorageSize& x) {
              const std::string t = "st_" + sanitize(x.tex->name);
              // WGSL reports array layers separately
              if (x.tex->dimension == TextureDimension::D2Array)
                return std::format("vec3u(textureDimensions({}), textureNumLayers({}))", t, t);
              return std::format("textureDimensions({})", t);
            },
        },
        e.node);
  }

  std::string sample(const ex::Sample& x) {
    const std::string name = sanitize(x.tex->name);
    const std::string head = std::format("t_{}, s_{}, {}", name, name, expr(*x.coord));
    const std::string layer = x.layer ? ", " + expr(*x.layer) : "";
    // level 0 comparison is valid in nonuniform control flow
    if (x.depthRef) return std::format("textureSampleCompareLevel({}, {})", head, expr(*x.depthRef));
    if (x.ddx)
      return std::format("textureSampleGrad({}{}, {}, {})", head, layer, expr(*x.ddx), expr(*x.ddy));
    const std::string level = x.level ? expr(*x.level) : "0.0";
    const std::string offset = x.offset ? ", " + expr(*x.offset) : "";
    return std::format("textureSampleLevel({}{}, {}{})", head, layer, level, offset);
  }

  std::string construct(const Expr& e, const std::vector<ExprPtr>& args) {
    const TypePtr& type = e.type;
    const Form form = lower(Target::WGSL, e).form;
    if (form == Form::Helper) return std::format("{}({})", mat3FromMat4Name(type), expr(*args[0]));
    if (form == Form::Inline && args[0]->type->kind == TypeKind::Matrix) {
      const std::string m = expr(*args[0]);
      return std::format("{}({}[0].xyz, {}[1].xyz, {}[2].xyz)", wgslType(type), m, m, m);
    }
    if (form == Form::Inline) {
      const std::string s = expr(*args[0]);
      std::vector<std::string> parts;
      for (int c = 0; c < type->cols; ++c)
        for (int r = 0; r < type->rows; ++r) parts.push_back(c == r ? s : literal(0.0, scalarType(type->scalar)));
      return std::format("{}({})", wgslType(type), join(parts, ", "));
    }
    return std::format("{}({})", wgslType(type), join(exprs(args), ", "));
  }

  /**
   * A const initializer cannot call a function, so the helper's formula is written out with each
   * operand printed once as a helper const; nested mods keep the text linear.
   */
  std::string constantMod(const TypePtr& type, const std::vector<std::string>& a) {
    std::array<std::string, 2> names;
    for (std::size_t i = 0; i < 2; ++i) {
      names[i] = std::format("moga_shared_{}", sharedCount_++);
      out.line(std::format("const {}: {} = {};", names[i], wgslType(type), a[i]));
    }
    return std::format("({0} - {1} * floor({0} / {1}))", names[0], names[1]);
  }

  std::string call(const Expr& e, const ex::Call& x) {
    const std::vector<std::string> a = exprs(x.args);
    const TypePtr& type = e.type;
    const Form form = lower(Target::WGSL, e).form;
    if (form == Form::Helper && x.fn == BuiltinFunction::Inverse)
      return std::format("{}({})", inverseName(type), a[0]);
    if (form == Form::Helper && inConstant_) return constantMod(type, a);
    if (form == Form::Helper) return std::format("moga_mod_{}({}, {})", wgslType(type), a[0], a[1]);
    if (form == Form::Inline) return std::format("({} - {} * floor({} / {}))", a[0], a[1], a[0], a[1]);
    if (isBitcast(x.fn)) return std::format("bitcast<{}>({})", wgslType(type), a[0]);
    return std::format("{}({})", callName(x.fn), join(a, ", "));
  }

  const AnalyzedProgram& program_;
  StageName stage_ = StageName::Vertex;
  /** Values printed once as locals, while the statement that declared them is printed. */
  std::unordered_map<const Expr*, std::string> shared_;
  /** True while the value of a module scope const is printed. */
  bool inConstant_ = false;
  /** Numbers the locals, so statements in one scope never declare a name twice. */
  int sharedCount_ = 0;
  /** Function whose body is being emitted; null inside an entry point. */
  const FunctionDecl* fn_ = nullptr;
  bool hasFragmentOutput_ = false;
  /** Constants emitted as functions. */
  std::set<const ConstantDecl*> runtimeConstants_;
};

} // namespace

std::string wgslType(const TypePtr& t) {
  switch (t->kind) {
  case TypeKind::Scalar:
    return typeName(t);
  case TypeKind::Vector:
    if (t->scalar == ScalarKind::Bool) return std::format("vec{}<bool>", t->size);
    return std::format("vec{}{}", t->size, scalarSuffix(t->scalar));
  case TypeKind::Matrix:
    return typeName(t);
  case TypeKind::Struct:
    return structName(t->name);
  case TypeKind::Array:
    return std::format("array<{}, {}>", wgslType(t->element), t->length);
  case TypeKind::Atomic:
    return typeName(t);
  case TypeKind::Void:
    break;
  }
  throw ShaderError(ErrorCode::Internal, "void is not a WGSL value type");
}

std::vector<Diagnostic> checkWGSL(const AnalyzedProgram& program) {
  const Reflection& r = program.reflection;
  std::vector<Diagnostic> gaps;
  const auto gap = [&](std::string message) {
    gaps.push_back({.code = ErrorCode::MissingFeature, .step = BuildStep::TargetCheck, .message = std::move(message)});
  };
  for (const FeatureUse& u : unsupported(r.features, webgpuCapabilities()))
    gap(std::format("{}: {} is not available in WebGPU", r.program, u.what));
  for (const BuiltinInput b : {BuiltinInput::DrawIndex, BuiltinInput::LocalInstanceIndex})
    if (r.usesVertexBuiltin(b))
      gap(std::format("{}: {} is GLSL-only; WebGPU variants use instanceIndex with firstInstance", r.program,
                      builtinInputName(b)));
  return gaps;
}

CompiledWGSL compileWGSL(const AnalyzedProgram& program) {
  // these errors are about the whole program, not the C++ line last built
  detail::clearUserLocation();
  if (const auto gaps = checkWGSL(program); !gaps.empty())
    throw ShaderError(gaps.front().code, gaps.front().message, std::nullopt);
  WGSLEmitter e(program);
  e.emit();
  if (program.reflection.compute) return {e.out.str(), "", "", e.out.mapping, "cs_main"};
  return {e.out.str(), "vs_main", "fs_main", e.out.mapping, ""};
}

} // namespace moga
