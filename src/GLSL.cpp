#include "BuiltinTable.hpp"
#include "Steps.hpp"
#include "Lower.hpp"
#include "TextEmitter.hpp"
#include "Walk.hpp"
#include "moga/Compile.hpp"

#include <cstdint>
#include <format>
#include <unordered_map>

namespace moga {

using namespace detail;

namespace {

std::string_view scalarName(ScalarKind k) {
  switch (k) {
  case ScalarKind::F32:
    return "float";
  case ScalarKind::I32:
    return "int";
  case ScalarKind::U32:
    return "uint";
  case ScalarKind::Bool:
    return "bool";
  }
  return "?";
}

std::string_view vectorPrefix(ScalarKind k) {
  switch (k) {
  case ScalarKind::F32:
    return "vec";
  case ScalarKind::I32:
    return "ivec";
  case ScalarKind::U32:
    return "uvec";
  case ScalarKind::Bool:
    return "bvec";
  }
  return "?";
}

std::string_view callName(BuiltinFunction fn) { return builtinSpec(fn).glsl; }

const char* vectorCompare(BinaryOp op) {
  switch (op) {
  case BinaryOp::Lt:
    return "lessThan";
  case BinaryOp::Le:
    return "lessThanEqual";
  case BinaryOp::Gt:
    return "greaterThan";
  case BinaryOp::Ge:
    return "greaterThanEqual";
  case BinaryOp::Eq:
    return "equal";
  case BinaryOp::Ne:
    return "notEqual";
  default:
    return nullptr;
  }
}

std::string builtinInput(BuiltinInput b) {
  switch (b) {
  case BuiltinInput::VertexIndex:
    return "uint(gl_VertexID)";
  case BuiltinInput::InstanceIndex:
    return std::format("uint(gl_InstanceID + {})", kGlslBaseInstanceUniform);
  case BuiltinInput::LocalInstanceIndex:
    return "uint(gl_InstanceID)";
  case BuiltinInput::DrawIndex:
    return std::format("uint(gl_DrawID + {})", kGlslDrawBaseUniform);
  case BuiltinInput::FragCoord:
    return "gl_FragCoord";
  case BuiltinInput::FrontFacing:
    return "gl_FrontFacing";
  }
  return "?";
}

/** `type name` or `type name[N]` (array suffix form used for fields). */
std::string declare(const TypePtr& t, std::string_view name) {
  if (t->kind == TypeKind::Array)
    return std::format("{} {}[{}]", glslType(t->element), name, t->length);
  return std::format("{} {}", glslType(t), name);
}

std::optional<std::string> zeroValue(const TypePtr& t);

/** The zero value of a struct field or array element, which must have one. */
std::string requiredZero(const TypePtr& t) {
  const std::optional<std::string> zero = zeroValue(t);
  if (!zero) throw ShaderError(ErrorCode::Internal, std::format("GLSL: {} has no zero value", typeName(t)));
  return *zero;
}

std::optional<std::string> zeroValue(const TypePtr& t) {
  switch (t->kind) {
  case TypeKind::Scalar:
    if (t->scalar == ScalarKind::Bool) return "false";
    return t->scalar == ScalarKind::F32 ? std::string("0.0") : glslType(t) + "(0)";
  case TypeKind::Vector:
    return glslType(t) + (t->scalar == ScalarKind::Bool ? "(false)" : "(0)");
  case TypeKind::Matrix:
    return glslType(t) + "(0.0)";
  case TypeKind::Struct: {
    // GLSL leaves locals undefined; WGSL zero fills them, so both start at zero.
    std::vector<std::string> fields;
    for (const StructField& f : t->fields) fields.push_back(requiredZero(f.type));
    return std::format("{}({})", glslType(t), join(fields, ", "));
  }
  case TypeKind::Array: {
    const std::vector<std::string> elements(static_cast<std::size_t>(t->length), requiredZero(t->element));
    return std::format("{}({})", glslType(t), join(elements, ", "));
  }
  case TypeKind::Atomic:
  case TypeKind::Void:
    break;
  }
  return std::nullopt;
}

std::string_view samplerPrefix(TextureSampleType s) {
  if (s == TextureSampleType::Uint) return "u";
  if (s == TextureSampleType::Sint) return "i";
  return "";
}

std::string_view samplerSuffix(TextureDimension d) {
  switch (d) {
  case TextureDimension::D2:
    return "sampler2D";
  case TextureDimension::D3:
    return "sampler3D";
  case TextureDimension::D2Array:
    return "sampler2DArray";
  case TextureDimension::Cube:
    return "samplerCube";
  }
  return "?";
}

std::string literal(const std::variant<double, bool>& value, const TypePtr& t) {
  if (const bool* b = std::get_if<bool>(&value)) return *b ? "true" : "false";
  const double v = std::get<double>(value);
  if (t->kind != TypeKind::Scalar) throw ShaderError(ErrorCode::Internal, "Literal must be scalar");
  std::string s;
  if (t->scalar == ScalarKind::F32) s = formatFloat(v);
  else if (t->scalar == ScalarKind::U32)
    s = std::format("{}u", static_cast<std::uint32_t>(static_cast<std::int64_t>(v)));
  else s = std::format("{}", static_cast<std::int32_t>(static_cast<std::int64_t>(v)));
  return v < 0 ? "(" + s + ")" : s;
}

class GLSLStageEmitter {
public:
  GLSLStageEmitter(const AnalyzedProgram& program, StageName stage, const GLSLOptions& options)
      : program_(program), stage_(stage), options_(options) {}

  TextEmitter out;

  void emit() {
    const bool vertex = stage_ == StageName::Vertex;
    preamble();
    for (const TypePtr& s : program_.structs) structDecl(*s);
    resources();
    if (vertex) drawUniforms();
    stageIO();
    out.line();
    for (const ConstantPtr& c : program_.constants) {
      // a reused value inside the constant is printed once, as a helper const
      std::vector<const Expr*> added;
      inConstant_ = true;
      for (const Expr* e : sharedSubexpressions(c->value)) {
        const std::string local = std::format("moga_shared_{}", sharedCount_++);
        out.line(std::format("const {} = {};", declare(e->type, local), exprBody(*e)));
        shared_.emplace(e, local);
        added.push_back(e);
      }
      const std::string value = expr(*c->value);
      inConstant_ = false;
      out.line(std::format("const {} {} = {};", glslType(c->type), constantName(c->name), value));
      for (const Expr* e : added) shared_.erase(e);
    }
    for (const Helper& h : program_.glslHelpers) helperDecl(h);
    for (const FunctionPtr& f : vertex ? program_.vertexFunctions : program_.fragmentFunctions)
      func(*f);
    if (vertex) clipFix();

    out.line("void main() {");
    out.indent();
    // point lists draw nothing unless the size is written; WebGPU points are one pixel
    if (vertex) out.line("gl_PointSize = 1.0;");
    zeroOutputs();
    block(vertex ? program_.ast->vertex : program_.ast->fragment);
    out.dedent();
    out.line("}");
  }

private:
  void preamble() {
    out.line("#version 300 es");
    if (stage_ == StageName::Vertex) vertexExtensions();
    if (stage_ == StageName::Fragment && program_.reflection.needs(DeviceFeature::DualSourceBlending))
      out.line("#extension GL_EXT_blend_func_extended : require");
    out.line("precision highp float;");
    out.line("precision highp int;");
    for (std::string_view s : {"sampler2D", "sampler3D", "sampler2DArray", "samplerCube",
                               "usampler2D", "isampler2D", "usampler3D", "sampler2DShadow"})
      out.line(std::format("precision highp {};", s));
    out.line();
  }

  void vertexExtensions() {
    const Reflection& r = program_.reflection;
    if (r.clipDistances > 0) out.line("#extension GL_ANGLE_clip_cull_distance : require");
    if (r.usesVertexBuiltin(BuiltinInput::DrawIndex)) out.line("#extension GL_ANGLE_multi_draw : require");
  }

  /** Base offsets added to gl_InstanceID and gl_DrawID. */
  void drawUniforms() {
    const Reflection& r = program_.reflection;
    if (r.usesVertexBuiltin(BuiltinInput::InstanceIndex))
      out.line(std::format("uniform int {};", kGlslBaseInstanceUniform));
    if (r.usesVertexBuiltin(BuiltinInput::DrawIndex))
      out.line(std::format("uniform int {};", kGlslDrawBaseUniform));
  }

  void clipFix() {
    out.line("vec4 moga_clipFix(vec4 p) {");
    out.indent();
    const char* z = options_.clipControl ? "p.z" : "2.0 * p.z - p.w";
    out.line(std::format("return vec4(p.x, {}, {}, p.w);", options_.flipY ? "-p.y" : "p.y", z));
    out.dedent();
    out.line("}");
  }

  void structDecl(const TypeInfo& s) {
    out.line(std::format("struct {} {{", structName(s.name)));
    out.indent();
    for (const StructField& f : s.fields) out.line(declare(f.type, sanitize(f.name)) + ";");
    out.dedent();
    out.line("};");
  }

  void resources() {
    const Reflection& r = program_.reflection;
    for (const ReflectedUniformBlock& ub : r.uniformBlocks) {
      out.line(std::format("layout(std140) uniform {} {{", ub.glslBlockName));
      out.indent();
      for (const StructField& f : ub.layout->structType->fields)
        out.line(declare(f.type, sanitize(f.name)) + ";");
      out.dedent();
      out.line(std::format("}} ub_{};", sanitize(ub.name)));
    }
    for (const ReflectedTexture& t : r.textures) {
      if (t.sampler == SamplerBinding::Comparison)
        out.line(std::format("uniform highp sampler2DShadow {};", t.glslName));
      else
        out.line(std::format("uniform highp {}{} {};", samplerPrefix(t.sampleType),
                             samplerSuffix(t.dimension), t.glslName));
    }
    for (const ReflectedTable& tb : r.tables) {
      out.line(std::format("uniform highp sampler2D {};", tb.glslName));
      tableFetch(tb);
    }
  }

  void tableFetch(const ReflectedTable& tb) {
    const int w = kTableTextureWidth;
    out.line(std::format("vec4 {}_fetch(uint record, int texel) {{", tb.glslName));
    out.indent();
    out.line(std::format("int t = int(record) * {} + texel;", tb.stride));
    out.line(std::format("return texelFetch({}, ivec2(t % {}, t / {}), 0);", tb.glslName, w, w));
    out.dedent();
    out.line("}");
    const std::string f = tb.glslName + "_fetch";
    out.line(std::format("mat4 {}Mat4(uint record, int texel) {{ return mat4({}(record, texel), "
                         "{}(record, texel + 1), {}(record, texel + 2), {}(record, texel + 3)); }}",
                         f, f, f, f, f));
  }

  /**
   * GLSL leaves an output that a path does not write undefined; WGSL zero
   * fills its output struct. Both start every output at zero.
   */
  void zeroOutputs() {
    const Reflection& r = program_.reflection;
    if (stage_ == StageName::Vertex) {
      for (const ReflectedVarying& v : r.varyings)
        out.line(std::format("v_{} = {};", sanitize(v.name), requiredZero(v.type)));
      for (int i = 0; i < r.clipDistances; ++i) out.line(std::format("gl_ClipDistance[{}] = 0.0;", i));
      return;
    }
    for (const ReflectedOutput& o : r.outputs)
      out.line(std::format("{} = {};", outputName(o.index, o.blendSrc), requiredZero(o.type)));
    if (r.usesFragDepth) out.line("gl_FragDepth = 0.0;");
  }

  void stageIO() {
    const Reflection& r = program_.reflection;
    const auto flat = [](Interpolation i) {
      return i == Interpolation::Flat || i == Interpolation::FlatEither ? "flat " : "";
    };
    if (stage_ == StageName::Vertex) {
      for (const ReflectedAttribute& a : r.attributes)
        out.line(std::format("layout(location = {}) in {} a_{};", a.location, glslType(a.type),
                             sanitize(a.name)));
      for (const ReflectedVarying& v : r.varyings)
        out.line(std::format("{}out {} v_{};", flat(v.interpolation), glslType(v.type),
                             sanitize(v.name)));
    } else {
      for (const ReflectedVarying& v : r.varyings)
        out.line(std::format("{}in {} v_{};", flat(v.interpolation), glslType(v.type),
                             sanitize(v.name)));
      for (const ReflectedOutput& o : r.outputs) {
        const std::string index = o.blendSrc == 1 ? ", index = 1" : "";
        out.line(std::format("layout(location = {}{}) out {} {};", o.index, index, glslType(o.type),
                             outputName(o.index, o.blendSrc)));
      }
    }
  }

  void func(const FunctionDecl& f) {
    std::vector<std::string> params;
    for (const LocalPtr& p : f.params) params.push_back(declare(p->type, localName(*p)));
    out.line(std::format("{} {}({}) {{", glslType(f.returnType), functionName(f), join(params, ", ")));
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

  /** Emits `header`, the indented `body` and the closing brace. */
  /** Above this many scalars and vectors, a zero value is written by loops. */
  static constexpr std::uint64_t kInlineZeroParts = 16;

  /** The scalars, vectors and matrices in a zero value of the type, saturated past the inline limit. */
  static std::uint64_t zeroParts(const TypePtr& t) {
    if (t->kind == TypeKind::Array)
      return std::min(kInlineZeroParts + 1, static_cast<std::uint64_t>(t->length) * zeroParts(t->element));
    if (t->kind != TypeKind::Struct) return 1;
    std::uint64_t n = 0;
    for (const StructField& f : t->fields) n = std::min(kInlineZeroParts + 1, n + zeroParts(f.type));
    return n;
  }

  /** Writes zero into target: a loop per large array, a statement per field of a large struct. */
  void zeroFill(const std::string& target, const TypePtr& t, const std::optional<DebugInfo>& d) {
    if (zeroParts(t) <= kInlineZeroParts) {
      out.line(std::format("{} = {};", target, requiredZero(t)), d);
      return;
    }
    if (t->kind == TypeKind::Struct) {
      for (const StructField& f : t->fields) zeroFill(std::format("{}.{}", target, sanitize(f.name)), f.type, d);
      return;
    }
    const std::string i = std::format("moga_i_{}", sharedCount_++);
    out.line(std::format("for (int {0} = 0; {0} < {1}; ++{0}) {{", i, t->length), d);
    out.indent();
    zeroFill(std::format("{}[{}]", target, i), t->element, d);
    out.dedent();
    out.line("}");
  }

  void nested(std::string_view header, const Block& body, const std::optional<DebugInfo>& debug) {
    out.line(header, debug);
    out.indent();
    block(body);
    out.dedent();
    out.line("}");
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
      out.line(std::format("{} = {};", declare(e->type, name), exprBody(*e)), s.debug);
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
              const std::string head = declare(x.v->type, name);
              // a large zero value would be one constructor argument per element: loops fill it instead
              if (!x.init && zeroParts(x.v->type) > kInlineZeroParts) {
                out.line(head + ";", d);
                zeroFill(name, x.v->type, d);
                return;
              }
              const auto init = x.init ? std::optional(expr(*x.init)) : zeroValue(x.v->type);
              out.line(init ? std::format("{} = {};", head, *init) : head + ";", d);
            },
            [&](const st::Assign& x) {
              const std::string op = x.op ? std::string(opText(*x.op)) : "";
              out.line(std::format("{} {}= {};", expr(*x.target), op, expr(*x.value)), d);
            },
            [&](const st::If& x) { ifStatement(x, d); },
            [&](const st::For& x) {
              const std::string i = localName(*x.v);
              nested(std::format("for ({} {} = {}; {} {} {}; {} += {}) {{", glslType(x.v->type), i,
                                 expr(*x.start), i, opText(x.condition), expr(*x.end), i,
                                 expr(*x.step)),
                     x.body, d);
            },
            [&](const st::While& x) {
              nested(std::format("while ({}) {{", expr(*x.cond)), x.body, d);
            },
            [&](const st::Switch& x) { switchStatement(x, d); },
            [&](const st::Loop& x) { nested("for (;;) {", x.body, d); },
            [&](const st::Break&) { out.line("break;", d); },
            [&](const st::Continue&) { out.line("continue;", d); },
            [&](const st::Discard&) { out.line("discard;", d); },
            [&](const st::Return& x) {
              out.line(fn_ == nullptr || !x.value ? "return;"
                                                  : std::format("return {};", expr(*x.value)),
                       d);
            },
            [&](const st::SetVarying& x) {
              // a varying the fragment stage never reads is not declared; a call in its value still runs
              if (program_.reflection.hasVarying(x.v->name))
                out.line(std::format("v_{} = {};", sanitize(x.v->name), expr(*x.value)), d);
              else if (hasSideEffects(*x.value))
                out.line(expr(*x.value) + ";", d);
            },
            [&](const st::SetOutput& x) {
              out.line(std::format("{} = {};", outputName(x.o->index, x.o->blendSrc), expr(*x.value)), d);
            },
            [&](const st::SetPosition& x) {
              out.line(std::format("gl_Position = moga_clipFix({});", expr(*x.value)), d);
            },
            [&](const st::SetFragDepth& x) {
              out.line(std::format("gl_FragDepth = {};", expr(*x.value)), d);
            },
            [&](const st::SetClipDistance& x) {
              out.line(std::format("gl_ClipDistance[{}] = {};", x.index, expr(*x.value)), d);
            },
            [&](const st::ExprStmt& x) { out.line(expr(*x.e) + ";", d); },
            // compileGLSL() rejects WebGPU only programs before any emission
            [&](const st::Barrier&) { unreachable(); },
            [&](const st::AtomicStore&) { unreachable(); },
            [&](const st::TextureStore&) { unreachable(); },
        },
        s.node);
  }

  /** Each case ends with break, so no case falls through to the next one. */
  void switchStatement(const st::Switch& s, const std::optional<DebugInfo>& debug) {
    out.line(std::format("switch ({}) {{", expr(*s.selector)), debug);
    out.indent();
    const auto clause = [&](const Block& body) {
      out.indent();
      block(body);
      out.line("break;");
      out.dedent();
      out.line("}");
    };
    for (const st::SwitchCase& c : s.cases) {
      for (std::size_t i = 0; i + 1 < c.values.size(); ++i)
        out.line(std::format("case {}:", expr(*c.values[i])));
      out.line(std::format("case {}: {{", expr(*c.values.back())));
      clause(c.body);
    }
    if (s.defaultBody) {
      out.line("default: {");
      clause(*s.defaultBody);
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
            [&](const ex::Constant& x) { return constantName(x.c->name); },
            [&](const ex::Attribute& x) { return "a_" + sanitize(x.a->name); },
            [&](const ex::Varying& x) { return "v_" + sanitize(x.v->name); },
            [&](const ex::Uniform& x) {
              return std::format("ub_{}.{}", sanitize(x.r->name), sanitize(x.field));
            },
            [&](const ex::LooseUniform& x) { return looseUniform(*x.u); },
            [&](const ex::Builtin& x) { return builtinInput(x.name); },
            [&](const ex::Member& x) { return std::format("{}.{}", expr(*x.base), sanitize(x.field)); },
            [&](const ex::Swizzle& x) {
              const std::string base = expr(*x.base);
              if (x.base->type->kind == TypeKind::Scalar)
                return x.comps.size() == 1 ? base : std::format("{}({})", glslType(e.type), base);
              return std::format("{}.{}", base, x.comps);
            },
            [&](const ex::Index& x) { return std::format("{}[{}]", expr(*x.base), expr(*x.index)); },
            [&](const ex::Unary& x) { return std::format("({}{})", opText(x.op), expr(*x.a)); },
            [&](const ex::Binary& x) {
              const std::string a = expr(*x.a);
              const std::string b = expr(*x.b);
              const char* cmp = vectorCompare(x.op);
              if (cmp != nullptr && x.a->type->kind == TypeKind::Vector)
                return std::format("{}({}, {})", cmp, a, b);
              return std::format("({} {} {})", a, opText(x.op), b);
            },
            [&](const ex::Call& x) { return call(e, x); },
            [&](const ex::UCall& x) {
              return std::format("{}({})", functionName(*x.fn), join(exprs(x.args), ", "));
            },
            [&](const ex::Construct& x) {
              return std::format("{}({})", glslType(e.type), join(exprs(x.args), ", "));
            },
            [&](const ex::Select& x) { return select(e, x); },
            [&](const ex::Sample& x) { return sample(x); },
            [&](const ex::Load& x) {
              std::string coord = expr(*x.coord);
              if (x.layer) coord = std::format("ivec3({}, {})", coord, expr(*x.layer));
              std::string fetch = std::format("texelFetch(t_{}, {}, {})", sanitize(x.tex->name),
                                              coord, intLevel(*x.level));
              return x.tex->sampleType == TextureSampleType::Depth ? fetch + ".x" : fetch;
            },
            [&](const ex::TexSize& x) {
              return std::format("textureSize(t_{}, {})", sanitize(x.tex->name), intLevel(*x.level));
            },
            [&](const ex::TableRead& x) { return tableRead(e, x); },
            [&](const ex::StorageElement&) -> std::string { unreachable(); },
            [&](const ex::StorageLength&) -> std::string { unreachable(); },
            [&](const ex::Workgroup&) -> std::string { unreachable(); },
            [&](const ex::Atomic&) -> std::string { unreachable(); },
            [&](const ex::StorageLoad&) -> std::string { unreachable(); },
            [&](const ex::StorageSize&) -> std::string { unreachable(); },
        },
        e.node);
  }

  /** GLSL helpers: an integer dot and a vector select that evaluate each operand once. */
  void helperDecl(const Helper& h) {
    const std::string t = glslType(h.type);
    if (h.kind == HelperKind::Smoothstep) {
      out.line(std::format("{0} moga_smoothstep_{1}({0} e0, {0} e1, {0} x) {{ {0} t = clamp((x - e0) / (e1 - e0), "
                           "0.0, 1.0); return t * t * (3.0 - 2.0 * t); }}",
                           t, typeTag(t)));
      return;
    }
    if (h.kind == HelperKind::IntegerDot) {
      std::vector<std::string> products;
      for (int i = 0; i < h.type->size; ++i) products.push_back(std::format("a.{0} * b.{0}", "xyzw"[i]));
      out.line(std::format("{} moga_dot_{}({} a, {} b) {{ return {}; }}", glslType(scalarType(h.type->scalar)),
                           t, t, t, join(products, " + ")));
    } else if (h.kind == HelperKind::ScalarSelect) {
      // function arguments are all evaluated, unlike the two branches of ?:
      out.line(std::format("{} moga_selectb_{}({} f, {} t, bool c) {{ return c ? t : f; }}", t,
                           typeTag(t), t, t));
    } else if (h.kind == HelperKind::VectorSelect) {
      std::vector<std::string> parts;
      for (int i = 0; i < h.type->size; ++i) parts.push_back(std::format("c.{0} ? t.{0} : f.{0}", "xyzw"[i]));
      out.line(std::format("{} moga_select_{}({} f, {} t, bvec{} c) {{ return {}({}); }}", t, t, t, t,
                           h.type->size, t, join(parts, ", ")));
    }
  }

  std::string select(const Expr& e, const ex::Select& x) {
    const std::string c = expr(*x.cond);
    const std::string t = expr(*x.t);
    const std::string f = expr(*x.f);
    const TypePtr& type = e.type;
    const Form form = lower(Target::GLSL, e).form;
    const bool vector = x.cond->type->kind == TypeKind::Vector;
    if (form == Form::Helper && !vector)
      return std::format("moga_selectb_{}({}, {}, {})", typeTag(glslType(type)), f, t, c);
    if (form == Form::Helper) return std::format("moga_select_{}({}, {}, {})", glslType(type), f, t, c);
    if (form == Form::Native && !vector) return std::format("({} ? {} : {})", c, t, f);
    if (form == Form::Native) return std::format("mix({}, {}, {})", f, t, c);
    std::vector<std::string> parts;
    for (int i = 0; i < type->size; ++i) {
      const char k = "xyzw"[i];
      parts.push_back(std::format("(({}).{} ? ({}).{} : ({}).{})", c, k, t, k, f, k));
    }
    return std::format("{}({})", glslType(type), join(parts, ", "));
  }

  /** texelFetch and textureSize take an int level, and GLSL does not convert uint to int. */
  std::string intLevel(const Expr& level) {
    const std::string l = expr(level);
    return level.type == u32() ? std::format("int({})", l) : l;
  }

  std::string looseUniform(const UniformDecl& u) {
    const UniformBlockResource* block = program_.blockOf(&u);
    if (block == nullptr)
      throw ShaderError(ErrorCode::Internal, std::format("GLSL: uniform {} is not part of the analyzed program", u.name));
    return std::format("ub_{}.{}", sanitize(block->name), sanitize(u.name));
  }

  std::string sample(const ex::Sample& x) {
    const std::string t = "t_" + sanitize(x.tex->name);
    std::string coord = expr(*x.coord);
    // a shadow lookup takes the reference as the last coordinate component
    if (x.depthRef) return std::format("textureLod({}, vec3({}, {}), 0.0)", t, coord, expr(*x.depthRef));
    if (x.layer) coord = std::format("vec3({}, float({}))", coord, expr(*x.layer));
    if (x.ddx) return std::format("textureGrad({}, {}, {}, {})", t, coord, expr(*x.ddx), expr(*x.ddy));
    // a depth texture is sampled as a plain sampler: an integer level becomes float, the result is .x
    const bool depth = x.tex->sampleType == TextureSampleType::Depth;
    std::string level = x.level ? expr(*x.level) : "0.0";
    if (depth) level = std::format("float({})", level);
    const std::string call =
        x.offset ? std::format("textureLodOffset({}, {}, {}, {})", t, coord, level, expr(*x.offset))
                 : std::format("textureLod({}, {}, {})", t, coord, level);
    return depth ? call + ".x" : call;
  }

  [[noreturn]] static void unreachable() {
    throw ShaderError(ErrorCode::Internal, "GLSL: a WebGPU only node reached the emitter");
  }

  /** Table records are whole texels: a mat4 field spans four consecutive texels. */
  std::string tableRead(const Expr& e, const ex::TableRead& x) {
    const ReflectedResource* r = program_.find(x.table.get());
    const auto* tb = r ? std::get_if<ReflectedTable>(r) : nullptr;
    if (tb == nullptr)
      throw ShaderError(ErrorCode::Internal, 
          std::format("GLSL: table {} is not part of the analyzed program", x.table->name));
    const int off = tb->fieldOffsets.at(x.field);
    const std::string idx = expr(*x.index);
    const Form form = lower(Target::GLSL, e).form;
    if (form == Form::Helper) return std::format("{}_fetchMat4({}, {})", tb->glslName, idx, off);
    if (form == Form::Inline) {
      std::vector<std::string> cols;
      for (int c = 0; c < 4; ++c) cols.push_back(std::format("{}_fetch({}, {})", tb->glslName, idx, off + c));
      return std::format("mat4({})", join(cols, ", "));
    }
    return std::format("{}_fetch({}, {})", tb->glslName, idx, off);
  }

  /**
   * A const initializer may only call builtins, so the helper's formula is written out:
   * t = clamp((x - e0) / (e1 - e0), 0, 1), then t * t * (3 - 2 * t). t is printed once as a
   * helper const, so nested calls keep the text linear.
   */
  std::string constantSmoothstep(const TypePtr& type, const std::vector<std::string>& a) {
    const std::string t = std::format("moga_shared_{}", sharedCount_++);
    out.line(std::format("const {} = clamp((({}) - ({})) / (({}) - ({})), 0.0, 1.0);", declare(type, t), a[2],
                         a[0], a[1], a[0]));
    return std::format("({0} * {0} * (3.0 - 2.0 * {0}))", t);
  }

  std::string call(const Expr& e, const ex::Call& x) {
    const std::vector<std::string> a = exprs(x.args);
    const Form form = lower(Target::GLSL, e).form;
    if (form == Form::Native) return std::format("{}({})", callName(x.fn), join(a, ", "));
    const TypePtr& t = x.args[0]->type;
    if (form == Form::Operand) return a[0];
    if (form == Form::Helper && x.fn == BuiltinFunction::Smoothstep && inConstant_)
      return constantSmoothstep(e.type, a);
    if (form == Form::Helper && x.fn == BuiltinFunction::Smoothstep)
      return std::format("moga_smoothstep_{}({}, {}, {})", typeTag(glslType(t)), a[0], a[1], a[2]);
    if (form == Form::Helper) return std::format("moga_dot_{}({}, {})", glslType(t), a[0], a[1]);
    if (form == Form::Inline && x.fn == BuiltinFunction::Saturate)
      return std::format("clamp({}, {}, {}(1.0))", a[0], zeroValue(t).value_or("0.0"), glslType(t));
    if (form == Form::Inline) {
      std::vector<std::string> products;
      for (int i = 0; i < t->size; ++i) {
        const char k = "xyzw"[i];
        products.push_back(std::format("({}).{} * ({}).{}", a[0], k, a[1], k));
      }
      return std::format("({})", join(products, " + "));
    }
    return std::format("{}({})", callName(x.fn), join(a, ", "));
  }

  const AnalyzedProgram& program_;
  StageName stage_;
  const GLSLOptions& options_;
  /** Values printed once as locals, while the statement that declared them is printed. */
  std::unordered_map<const Expr*, std::string> shared_;
  /** Numbers the locals, so statements in one scope never declare a name twice. */
  int sharedCount_ = 0;
  /** True while the value of a module scope const is printed. */
  bool inConstant_ = false;
  /** Function whose body is being emitted; null inside `main`. */
  const FunctionDecl* fn_ = nullptr;
};

} // namespace

std::string glslType(const TypePtr& t) {
  switch (t->kind) {
  case TypeKind::Scalar:
    return std::string(scalarName(t->scalar));
  case TypeKind::Vector:
    return std::format("{}{}", vectorPrefix(t->scalar), t->size);
  case TypeKind::Matrix:
    return t->cols == t->rows ? std::format("mat{}", t->cols)
                              : std::format("mat{}x{}", t->cols, t->rows);
  case TypeKind::Struct:
    return structName(t->name);
  case TypeKind::Array:
    return std::format("{}[{}]", glslType(t->element), t->length);
  case TypeKind::Atomic:
    throw ShaderError(ErrorCode::MissingFeature, "GLSL has no atomic types; atomics need WebGPU");
  case TypeKind::Void:
    return "void";
  }
  return "?";
}

std::vector<Diagnostic> checkGLSL(const AnalyzedProgram& program, const GLSLOptions& options) {
  const Reflection& r = program.reflection;
  std::vector<Diagnostic> gaps;
  const auto gap = [&](std::string message) {
    gaps.push_back({.code = ErrorCode::MissingFeature, .step = BuildStep::TargetCheck, .message = std::move(message)});
  };
  for (const FeatureUse& u :
       unsupported(r.features, webgl2Capabilities(options.firstVertexConvention, options.blendFuncExtended))) {
    if (u.feature == Feature::FlatFirstVertex)
      gap(std::format("{}: {} takes the first vertex as in WebGPU; enable the first vertex convention "
                      "(WEBGL_provoking_vertex) and set GLSLOptions::firstVertexConvention, or use "
                      "Interpolation::FlatEither",
                      r.program, u.what));
    else
      gap(std::format("{}: {} needs WebGPU; WebGL2 cannot run it", r.program, u.what));
  }
  if (r.clipDistances > 0 && !options.clipDistanceExtension)
    gap(std::format("{}: ClipDistance needs WEBGL_clip_cull_distance (GLSLOptions::clipDistanceExtension)",
                    r.program));
  if (r.usesVertexBuiltin(BuiltinInput::DrawIndex) && !options.multiDraw)
    gap(std::format("{}: drawIndex needs WEBGL_multi_draw (GLSLOptions::multiDraw)", r.program));
  return gaps;
}

CompiledGLSL compileGLSL(const AnalyzedProgram& program, const GLSLOptions& options) {
  // these errors are about the whole program, not the C++ line last built
  detail::clearUserLocation();
  if (const auto gaps = checkGLSL(program, options); !gaps.empty())
    throw ShaderError(gaps.front().code, gaps.front().message, std::nullopt);
  GLSLStageEmitter vs(program, StageName::Vertex, options);
  vs.emit();
  GLSLStageEmitter fs(program, StageName::Fragment, options);
  fs.emit();
  const Reflection& r = program.reflection;
  CompiledGLSL out{vs.out.str(), fs.out.str(), vs.out.mapping, fs.out.mapping, {}, {}};
  if (r.usesVertexBuiltin(BuiltinInput::InstanceIndex))
    out.baseInstanceUniform = std::string(kGlslBaseInstanceUniform);
  if (r.usesVertexBuiltin(BuiltinInput::DrawIndex))
    out.drawBaseUniform = std::string(kGlslDrawBaseUniform);
  return out;
}

} // namespace moga
