#pragma once

// Program AST. Expressions are immutable and shared; declarations and
// resources are compared by identity (pointer).

#include "moga/Types.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace moga {

// ------------------------------------------------------------------ resources

/** Update frequency decides the backend binding group / slot range. */
enum class ResourceFrequency { Frame, Object, Material };
/** The kind of a bound resource. */
enum class ResourceKind { UniformBlock, Table, Texture, StorageBuffer, StorageTexture };

/**
 * The dimension of a texture: 2D, 3D, 2D array or cube. A cube has six
 * square faces, sampled with a vec3 direction (no texel loads).
 */
enum class TextureDimension { D2, D3, D2Array, Cube };
/** The texel type of a texture, as in GPUTextureSampleType. Uint and Sint textures cannot be sampled; use load(). */
enum class TextureSampleType { Float, UnfilterableFloat, Uint, Sint, Depth };

/** Access of a storage buffer or a storage texture from the shader. */
enum class StorageAccess { Read, Write, ReadWrite };

/**
 * Storage texture formats. Some need a device feature; see
 * storageFormatFeatures() in Capabilities.hpp.
 */
enum class TexelFormat {
  Rgba8Unorm, Rgba8Snorm, Rgba8Uint, Rgba8Sint, Bgra8Unorm, Rgba16Float, Rgba16Uint, Rgba16Sint,
  Rgba32Float, Rgba32Uint, Rgba32Sint, R32Float, R32Uint, R32Sint, Rg32Float, Rg32Uint, Rg32Sint,
  // texture-formats-tier1
  R8Unorm, R8Snorm, R8Uint, R8Sint, R16Unorm, R16Snorm, R16Uint, R16Sint, R16Float, Rg8Unorm,
  Rg8Snorm, Rg8Uint, Rg8Sint, Rg16Unorm, Rg16Snorm, Rg16Uint, Rg16Sint, Rg16Float, Rgb10a2Uint,
  Rgb10a2Unorm, Rg11b10Ufloat, Rgba16Unorm, Rgba16Snorm
};

/** Returns the WGSL name of a texel format, for example "rgba8unorm". */
std::string_view texelFormatName(TexelFormat f);
/** Returns the component kind that a format reads and writes: f32, u32 or i32. */
ScalarKind texelFormatScalar(TexelFormat f);

/** The common part of a bound resource. Resources compare by identity. */
struct Resource {
  /** The kind of resource. It tells the derived type. */
  ResourceKind kind;
  /** The name of the resource. */
  std::string name;
  /** The update frequency. It selects the bind group. */
  ResourceFrequency frequency;

protected:
  Resource(ResourceKind k, std::string n, ResourceFrequency f)
      : kind(k), name(std::move(n)), frequency(f) {}
};

/** A uniform block. */
struct UniformBlockResource : Resource {
  /** The struct of the block fields. */
  TypePtr structType;
  /** Makes the resource. Use uniformBlock() instead. */
  UniformBlockResource(std::string n, TypePtr s, ResourceFrequency f)
      : Resource(ResourceKind::UniformBlock, std::move(n), f), structType(std::move(s)) {}
};

/** A sampled or loaded texture. */
struct TextureResource : Resource {
  TextureDimension dimension;
  TextureSampleType sampleType;
  /** Whether a sampler is paired (WGSL emits a separate sampler binding). */
  bool sampled;
  /** The sampler compares depth values (a shadow sampler). */
  bool comparison;
  /** The texels are 32 bit floats and the sampler filters them (float32-filterable). */
  bool float32Filterable;
  /** Makes the resource. Use texture() instead. */
  TextureResource(std::string n, TextureDimension d, TextureSampleType st, ResourceFrequency f,
                  bool s, bool c = false, bool f32Filterable = false)
      : Resource(ResourceKind::Texture, std::move(n), f), dimension(d), sampleType(st),
        sampled(s), comparison(c), float32Filterable(f32Filterable) {}
};

/**
 * Indexed per instance records ("object table"). WGSL lowers to a read only
 * storage buffer; GLSL lowers to an RGBA32F texture with generated fetches.
 * Every field must be vec4f or mat4f so records are whole texels.
 */
struct TableResource : Resource {
  /** The struct of one record. */
  TypePtr structType;
  /** Texels (vec4) per record. */
  int stride;
  /** Makes the resource. Use table() instead. */
  TableResource(std::string n, TypePtr s, ResourceFrequency f, int st)
      : Resource(ResourceKind::Table, std::move(n), f), structType(std::move(s)), stride(st) {}
};

/**
 * A runtime sized array of elements in the storage address space. WebGPU only.
 * Read/write buffers are not valid in the vertex stage.
 */
struct StorageBufferResource : Resource {
  /** The type of one element. */
  TypePtr elementType;
  /** Read or ReadWrite. */
  StorageAccess access;
  /** Makes the resource. Use storage() instead. */
  StorageBufferResource(std::string n, TypePtr e, StorageAccess a, ResourceFrequency f)
      : Resource(ResourceKind::StorageBuffer, std::move(n), f), elementType(std::move(e)),
        access(a) {}
};

/** A texture that the shader writes or reads without a sampler. WebGPU only. */
struct StorageTextureResource : Resource {
  /** D2, D3 or D2Array. */
  TextureDimension dimension;
  /** The texel format. */
  TexelFormat format;
  /** Read, Write or ReadWrite. */
  StorageAccess access;
  /** Makes the resource. Use storageTexture() instead. */
  StorageTextureResource(std::string n, TextureDimension d, TexelFormat fmt, StorageAccess a,
                         ResourceFrequency f)
      : Resource(ResourceKind::StorageTexture, std::move(n), f), dimension(d), format(fmt),
        access(a) {}
};

/** A shared handle to a resource of any kind. */
using ResourcePtr = std::shared_ptr<const Resource>;
/** A shared handle to a storage buffer resource. */
using StorageBufferPtr = std::shared_ptr<const StorageBufferResource>;
/** A shared handle to a storage texture resource. */
using StorageTexturePtr = std::shared_ptr<const StorageTextureResource>;
/** A shared handle to a uniform block resource. */
using UniformBlockPtr = std::shared_ptr<const UniformBlockResource>;
/** A shared handle to a texture resource. */
using TexturePtr = std::shared_ptr<const TextureResource>;
/** A shared handle to a table resource. */
using TablePtr = std::shared_ptr<const TableResource>;

/** How a vertex attribute advances: once per vertex or once per instance. */
enum class VertexStep { Vertex, Instance };
/**
 * How a varying is interpolated across a primitive.
 * - Perspective: perspective correct (the default).
 * - Linear: linear in screen space. WebGPU only (GLSL ES 3.00 has no noperspective).
 * - Flat: the value of the first vertex. WebGL2 must switch to the first
 *   vertex convention (WEBGL_provoking_vertex); see GLSLOptions::firstVertexConvention.
 * - FlatEither: the value of any one vertex. Use it for values that are the
 *   same on every vertex of a primitive (instance or object ids); it needs no extension.
 */
enum class Interpolation { Perspective, Linear, Flat, FlatEither };

/** A vertex attribute declaration. */
struct AttributeDecl {
  std::string name;
  TypePtr type;
  VertexStep step;
};

/** A varying declaration. */
struct VaryingDecl {
  std::string name;
  TypePtr type;
  Interpolation interpolation;
};

/** A fragment output declaration. */
struct OutputDecl {
  /** The location of the output (the color attachment index). */
  int index;
  TypePtr type;
  /** The blend source (0 or 1) of a dual source blending output at location 0. */
  std::optional<int> blendSrc = std::nullopt;
};

/** Returns the name of an output in both backends: o_0, or o_0_src1 for blend source 1. */
std::string outputName(int index, std::optional<int> blendSrc);

struct Expr;
/** A shared handle to an immutable expression node. */
using ExprPtr = std::shared_ptr<const Expr>;

/** A named constant. Its value is built from literals only. */
struct ConstantDecl {
  std::string name;
  TypePtr type;
  ExprPtr value;
};

/**
 * A single uniform. Analysis packs the uniforms of each frequency into one
 * generated uniform block.
 */
struct UniformDecl {
  std::string name;
  TypePtr type;
  ResourceFrequency frequency;
};

/** An array shared by the invocations of one workgroup. Compute stage only. */
struct WorkgroupDecl {
  std::string name;
  /** The array type. */
  TypePtr type;
};

/** A local variable or constant, or a function parameter. */
struct LocalVar {
  /** A unique id. */
  int id;
  /** The name that the user gave. */
  std::string name;
  TypePtr type;
  /** True for a variable that can be assigned (Var()). */
  bool mutableVar;
};

/** A shared handle to an attribute declaration. */
using AttributePtr = std::shared_ptr<const AttributeDecl>;
/** A shared handle to a varying declaration. */
using VaryingPtr = std::shared_ptr<const VaryingDecl>;
/** A shared handle to an output declaration. */
using OutputPtr = std::shared_ptr<const OutputDecl>;
/** A shared handle to a local variable. */
using LocalPtr = std::shared_ptr<const LocalVar>;
/** A shared handle to a named constant. */
using ConstantPtr = std::shared_ptr<const ConstantDecl>;
/** A shared handle to a single uniform. */
using UniformPtr = std::shared_ptr<const UniformDecl>;
/** A shared handle to a workgroup array declaration. */
using WorkgroupPtr = std::shared_ptr<const WorkgroupDecl>;

/** A builtin stage input. The DSL function of the same name reads it. */
enum class BuiltinInput {
  /** Vertex stage: vertexIndex(). */
  VertexIndex,
  /** Vertex stage: instanceIndex(). */
  InstanceIndex,
  /** Vertex stage, GLSL only: localInstanceIndex(). */
  LocalInstanceIndex,
  /** Vertex stage, GLSL with WEBGL_multi_draw only: drawIndex(). */
  DrawIndex,
  /** Fragment stage: fragCoord(). */
  FragCoord,
  /** Fragment stage: frontFacing(). */
  FrontFacing,
  /** Compute stage: globalId(). */
  GlobalId,
  /** Compute stage: localId(). */
  LocalId,
  /** Compute stage: workgroupId(). */
  WorkgroupId,
  /** Compute stage: numWorkgroups(). */
  NumWorkgroups,
  /** Compute stage: localIndex(). */
  LocalIndex,
  /** Compute and fragment stages: subgroupSize(). */
  SubgroupSize,
  /** Compute and fragment stages: subgroupInvocationId(). */
  SubgroupInvocationId,
  /** Fragment stage, WebGPU only (primitive-index). */
  PrimitiveIndex
};

/** Returns true for the compute builtins. */
bool isComputeBuiltin(BuiltinInput b);
/** Returns true for the subgroup builtins. */
bool isSubgroupBuiltin(BuiltinInput b);

/** A builtin function. builtinName() gives its DSL name. */
enum class BuiltinFunction {
  Abs, Sign, Floor, Ceil, Fract, Round, Trunc, Sqrt, InverseSqrt, Exp, Exp2, Log, Log2,
  Pow, Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Min, Max, Clamp, Saturate, Mix, Step,
  Smoothstep, Length, Distance, Dot, Cross, Normalize, Reflect, FaceForward, Mod, Dpdx,
  Dpdy, Fwidth, Transpose, Determinant, Any, All, FloatBitsToUint, UintBitsToFloat,
  FloatBitsToInt, IntBitsToFloat, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Radians, Degrees,
  Refract, Inverse, PackHalf2x16, PackSnorm2x16, PackUnorm2x16, UnpackHalf2x16,
  UnpackSnorm2x16, UnpackUnorm2x16, SubgroupAdd, SubgroupMul, SubgroupMin, SubgroupMax,
  SubgroupAnd, SubgroupOr, SubgroupXor, SubgroupAll, SubgroupAny, SubgroupBroadcastFirst,
  SubgroupBroadcast, SubgroupShuffle, SubgroupElect, SubgroupBallot
};

/** Returns true for the subgroup functions. */
bool isSubgroupFunction(BuiltinFunction fn);

/** Atomic read modify write operations. Each one returns the old value. */
enum class AtomicOp { Load, Add, Sub, Max, Min, And, Or, Xor, Exchange };

/** Returns the WGSL name, for example "atomicAdd". */
std::string_view atomicOpName(AtomicOp op);

/** The kind of a barrier: workgroupBarrier(), storageBarrier() or textureBarrier(). */
enum class BarrierKind { Workgroup, Storage, Texture };

/** Returns the DSL name of a builtin function, for example "inverseSqrt". */
std::string_view builtinName(BuiltinFunction fn);
/** Returns the DSL name of a builtin input, for example "vertexIndex". */
std::string_view builtinInputName(BuiltinInput b);

/** A binary operator. And and Or are logical; the Bit operators are bitwise. */
enum class BinaryOp { Add, Sub, Mul, Div, Rem, Lt, Le, Gt, Ge, Eq, Ne, And, Or, BitAnd,
                      BitOr, BitXor, Shl, Shr };
/** A unary operator: negation, logical not or bitwise not. */
enum class UnaryOp { Neg, Not, BitNot };

/** Returns the operator symbol, for example "+". */
std::string_view opText(BinaryOp op);
/** Returns the operator symbol, for example "-". */
std::string_view opText(UnaryOp op);

// ------------------------------------------------------------------ expressions

struct Stmt;
struct FunctionDecl;
/** A shared handle to a statement. */
using StmtPtr = std::shared_ptr<Stmt>;
/** A list of statements: a stage body, a function body or a branch. */
using Block = std::vector<StmtPtr>;
/** A shared handle to a user function declaration. */
using FunctionPtr = std::shared_ptr<const FunctionDecl>;

/** The expression node kinds. */
namespace ex {
/** A number or bool literal. The expression type gives the number type. */
struct Lit { std::variant<double, bool> value; };
/** Reads a local variable, a constant or a parameter. */
struct Local { LocalPtr v; };
/** Reads a named constant. */
struct Constant { ConstantPtr c; };
/** Reads a vertex attribute. */
struct Attribute { AttributePtr a; };
/** Reads a varying in the fragment stage. */
struct Varying { VaryingPtr v; };
/** Reads a field of a uniform block. */
struct Uniform { UniformBlockPtr r; std::string field; };
/** Reads a single uniform. */
struct LooseUniform { UniformPtr u; };
/** An element of a storage buffer. */
struct StorageElement { StorageBufferPtr buffer; ExprPtr index; };
/** The number of elements of a storage buffer (u32). */
struct StorageLength { StorageBufferPtr buffer; };
/** A workgroup array. */
struct Workgroup { WorkgroupPtr w; };
/** `target` is a storage element or a workgroup element of atomic type. */
struct Atomic { AtomicOp op; ExprPtr target; ExprPtr value; };
/** Reads one texel of a storage texture. `layer` is null except for 2D arrays. */
struct StorageLoad { StorageTexturePtr tex; ExprPtr coord; ExprPtr layer; };
/** The size of a storage texture in texels. */
struct StorageSize { StorageTexturePtr tex; };
/** Reads a builtin stage input. */
struct Builtin { BuiltinInput name; };
/** Reads a struct member. */
struct Member { ExprPtr base; std::string field; };
/** Selects vector components, for example "xzy". */
struct Swizzle { ExprPtr base; std::string comps; };
/** An array element, a matrix column or a vector component. */
struct Index { ExprPtr base; ExprPtr index; };
/** A unary operation. */
struct Unary { UnaryOp op; ExprPtr a; };
/** A binary operation. */
struct Binary { BinaryOp op; ExprPtr a; ExprPtr b; };
/** A call to a builtin function. */
struct Call { BuiltinFunction fn; std::vector<ExprPtr> args; };
/** A call to a user function. */
struct UCall { FunctionPtr fn; std::vector<ExprPtr> args; };
/** Makes a value of the expression type from its arguments. */
struct Construct { std::vector<ExprPtr> args; };
/** `t` when `cond` is true, else `f`. Both are evaluated. */
struct Select { ExprPtr cond; ExprPtr t; ExprPtr f; };
/**
 * A filtered sample. It uses an explicit level, or gradients (ddx, ddy) when
 * they are set. `offset` is a constant texel offset; `depthRef` makes it a
 * depth comparison.
 */
struct Sample {
  TexturePtr tex;
  ExprPtr coord;
  ExprPtr level;
  ExprPtr layer;
  ExprPtr ddx = nullptr;
  ExprPtr ddy = nullptr;
  ExprPtr offset = nullptr;
  ExprPtr depthRef = nullptr;
};
/** Reads one texel without filtering. */
struct Load { TexturePtr tex; ExprPtr coord; ExprPtr level; ExprPtr layer; };
/** The size of a texture at a level, in texels. */
struct TexSize { TexturePtr tex; ExprPtr level; };
/** Reads a field of one table record. */
struct TableRead { TablePtr table; ExprPtr index; std::string field; };
} // namespace ex

/** One expression node of any kind. */
using ExprNode = std::variant<ex::Lit, ex::Local, ex::Constant, ex::Attribute, ex::Varying,
                              ex::Uniform, ex::LooseUniform, ex::Builtin, ex::Member, ex::Swizzle, ex::Index, ex::Unary,
                              ex::Binary, ex::Call, ex::UCall, ex::Construct, ex::Select,
                              ex::Sample, ex::Load, ex::TexSize, ex::TableRead,
                              ex::StorageElement, ex::StorageLength, ex::Workgroup, ex::Atomic,
                              ex::StorageLoad, ex::StorageSize>;

/** An immutable expression: its node, its type and facts about its operands. */
struct Expr {
  /** A unique id. */
  int id;
  /** The type of the value. */
  TypePtr type;
  /** The node. */
  ExprNode node;
  // Facts about the node and its operands, set once by makeExpr(), so the
  // passes that need them never walk a shared graph again.
  /** A user call or an atomic is in it; read it with hasSideEffects(). */
  bool sideEffects = false;
  /** Every leaf is a literal or a named constant (a const expression needs this). */
  bool constantLeaves = false;
  /** A derivative or a user call is in it (see the uniformity check). */
  bool needsUniformity = false;
  /** It reads memory that a call or an atomic can change: writable storage or workgroup memory. */
  bool readsMutableMemory = false;
  /** The number of nodes when shared operands are written out again, saturated at kTreeSizeLimit. */
  std::uint32_t treeSize = 1;
  /**
   * It stands for, or uses, a value that an error could not make. Errors about
   * it are not reported again; a build with one never reaches analysis.
   */
  bool poisoned = false;
};

/** Where treeSize stops counting. */
inline constexpr std::uint32_t kTreeSizeLimit = 1u << 20;

/** Makes an expression node with a new id. */
ExprPtr makeExpr(ExprNode node, TypePtr type);

/** Makes a poisoned node of a type: a placeholder for a value that an error could not make. */
ExprPtr makePoison(TypePtr type);

/**
 * Returns true when evaluating the expression can change state: it calls a
 * user function (which can write storage) or runs an atomic. Code that would
 * repeat such an expression must evaluate it once instead.
 */
bool hasSideEffects(const Expr& e);

// ------------------------------------------------------------------ statements

/** The statement node kinds. */
namespace st {
/** Declares a local. `init` is null for a variable set to zero. */
struct Decl { LocalPtr v; ExprPtr init; };
/** `target = value`, or `target op= value` when `op` is set (the target is evaluated once). */
struct Assign { ExprPtr target; ExprPtr value; std::optional<BinaryOp> op = std::nullopt; };
/** `elseBody` is attached later by an If chain. */
struct If { ExprPtr cond; Block body; std::optional<Block> elseBody; };
/** `for (v = start; v <condition> end; v += step)`. */
struct For {
  LocalPtr v;
  ExprPtr start;
  ExprPtr end;
  ExprPtr step;
  Block body;
  BinaryOp condition = BinaryOp::Lt;
};
/** Runs `body` while `cond` is true. */
struct While { ExprPtr cond; Block body; };
/** Runs `body` until a Break or a Return. */
struct Loop { Block body; };
/** One case of a switch. It never falls through to the next case. */
struct SwitchCase { std::vector<ExprPtr> values; Block body; };
/** Cases are added later by a Switch chain. */
struct Switch { ExprPtr selector; std::vector<SwitchCase> cases; std::optional<Block> defaultBody; };
/** Stops the loop or leaves the switch case. */
struct Break {};
/** Starts the next iteration of the loop. */
struct Continue {};
/** Returns from the function or stage. `value` is null for no value. */
struct Return { ExprPtr value; };
/** Discards the fragment. */
struct Discard {};
/** Writes a varying in the vertex stage. */
struct SetVarying { VaryingPtr v; ExprPtr value; };
/** Writes a fragment output. */
struct SetOutput { OutputPtr o; ExprPtr value; };
/** Writes the clip space position. */
struct SetPosition { ExprPtr value; };
/** Writes the fragment depth. */
struct SetFragDepth { ExprPtr value; };
/** Writes one clip distance. */
struct SetClipDistance { int index; ExprPtr value; };
/** Evaluates an expression and ignores its value, for example a void call. */
struct ExprStmt { ExprPtr e; };
/** A workgroup, storage or texture barrier. */
struct Barrier { BarrierKind kind; };
/** Writes an atomic value. */
struct AtomicStore { ExprPtr target; ExprPtr value; };
/** Writes one texel of a storage texture. */
struct TextureStore { StorageTexturePtr tex; ExprPtr coord; ExprPtr layer; ExprPtr value; };
} // namespace st

/** One statement node of any kind. */
using StmtNode = std::variant<st::Decl, st::Assign, st::If, st::For, st::While, st::Switch,
                              st::Loop, st::Break,
                              st::Continue, st::Return, st::Discard, st::SetVarying,
                              st::SetOutput, st::SetPosition, st::SetFragDepth,
                              st::SetClipDistance, st::ExprStmt, st::Barrier,
                              st::AtomicStore, st::TextureStore>;

/** A statement and the C++ location that recorded it. */
struct Stmt {
  /** The node. */
  StmtNode node;
  /** The C++ location; set only when source locations are on. */
  std::optional<DebugInfo> debug;
};

/** A user function declaration, made by fn(). */
struct FunctionDecl {
  std::string name;
  /** The parameters, in order. */
  std::vector<LocalPtr> params;
  /** The return type, or voidType(). */
  TypePtr returnType;
  Block body;
};

/** A shader stage. */
enum class StageName { Vertex, Fragment, Compute };

/** A built program before backend compilation. */
struct ProgramAST {
  /** The program name. */
  std::string name;
  /** Variant key that produced this program (stable, deterministic). */
  std::string variantKey;
  /** The vertex stage body. */
  Block vertex;
  /** The fragment stage body. */
  Block fragment;
  /** A compute program has only a compute body. WebGPU only. */
  bool compute = false;
  /** The compute stage body. */
  Block computeBody;
  /** The workgroup size of a compute program. */
  std::array<std::uint32_t, 3> workgroupSize{1, 1, 1};
  /** The subgroup size of a compute program (subgroup-size-control). */
  std::optional<std::uint32_t> subgroupSize;
};

/** Combines lambdas into one visitor for std::visit. */
template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};

} // namespace moga
