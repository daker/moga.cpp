#pragma once

// GLSL ES 3.00 (WebGL2) and WGSL (WebGPU) backends.

#include "moga/Analyze.hpp"
#include "moga/Diagnostic.hpp"

#include <optional>
#include <string>
#include <vector>

namespace moga {

/** Maps a generated source line (1 based) back to the DSL statement that produced it. */
struct LineMapping {
  /** The 1 based line in the generated code. */
  int line;
  /** The C++ location of the statement. */
  DebugInfo debug;
};

/** Options of the GLSL backend: the WebGL2 extensions and conventions that the renderer uses. */
struct GLSLOptions {
  /**
   * Renders offscreen targets with clip space Y negated, so framebuffer memory,
   * fragCoord, derivatives and readback rows match WebGPU. The renderer sets
   * frontFace=CW and flips once when it presents.
   */
  bool flipY = true;
  /** EXT_clip_control (zero to one depth) is active; otherwise depth is remapped. */
  bool clipControl = false;
  /** WEBGL_clip_cull_distance is available (required if the program uses ClipDistance). */
  bool clipDistanceExtension = false;
  /** WEBGL_multi_draw is available (required if the program uses drawIndex). */
  bool multiDraw = false;
  /**
   * The renderer switched WebGL2 to the first vertex convention
   * (WEBGL_provoking_vertex, provokingVertexWEBGL(FIRST_VERTEX_CONVENTION_WEBGL)).
   * Required if the program has an Interpolation::Flat varying; WebGL2 uses
   * the last vertex by default, WebGPU the first.
   */
  bool firstVertexConvention = false;
  /**
   * WEBGL_blend_func_extended is enabled, so the fragment shader can write
   * the two outputs of dual source blending (blendOutput()).
   */
  bool blendFuncExtended = false;
};

/** The name of the GLSL uniform that holds the first instance of the draw. */
inline constexpr std::string_view kGlslBaseInstanceUniform = "moga_BaseInstance";
/** The name of the GLSL uniform that holds the draw base of drawIndex(). */
inline constexpr std::string_view kGlslDrawBaseUniform = "moga_DrawBase";

/** The GLSL ES 3.00 code of a render program. */
struct CompiledGLSL {
  /** The vertex shader source. */
  std::string vertex;
  /** The fragment shader source. */
  std::string fragment;
  /** Maps the lines of the vertex shader to C++ statements. */
  std::vector<LineMapping> vertexLineMap;
  /** Maps the lines of the fragment shader to C++ statements. */
  std::vector<LineMapping> fragmentLineMap;
  /** Name of the per draw base instance uniform when instanceIndex is used. */
  std::optional<std::string> baseInstanceUniform;
  /** Name of the per call draw base uniform when drawIndex is used. */
  std::optional<std::string> drawBaseUniform;
};

/** The WGSL code of a program. One module holds all entry points. */
struct CompiledWGSL {
  /** The WGSL module source. */
  std::string code;
  /** The vertex entry point name ("vs_main"); empty for a compute program. */
  std::string vertexEntry;
  /** The fragment entry point name ("fs_main"); empty for a compute program. */
  std::string fragmentEntry;
  /** Maps the lines of the module to C++ statements. */
  std::vector<LineMapping> lineMap;
  /** Entry point of a compute program; empty for a render program. */
  std::string computeEntry;
};

/** Returns the GLSL name of a type. */
std::string glslType(const TypePtr& t);
/** Returns the WGSL name of a type. Throws for void. */
std::string wgslType(const TypePtr& t);

} // namespace moga
