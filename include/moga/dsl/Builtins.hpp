#pragma once

// Builtin stage inputs and stage outputs: position, depth and clip distances.

#include "moga/dsl/Value.hpp"

namespace moga {

/** Returns the vertex index (u32). Vertex stage only. */
Val vertexIndex();
/** Returns the instance index (u32). It includes the first instance of the draw. Vertex stage only. */
Val instanceIndex();
/** Returns the instance index in the draw, without the first instance (u32). Vertex stage, GLSL only. */
Val localInstanceIndex();
/** Returns the subdraw index plus the draw base (u32). Vertex stage, GLSL with WEBGL_multi_draw only. */
Val drawIndex();
/** Returns the framebuffer position (vec4f). The origin is top left and pixel centers are at .5. Fragment stage only. */
Val fragCoord();
/** Returns true when the fragment is on a front face. Fragment stage only. */
Val frontFacing();
/**
 * Returns the index of the primitive that the fragment belongs to (u32),
 * counted from the start of the draw. Fragment stage only. WebGPU only; needs
 * the primitive-index device feature.
 */
Val primitiveIndex();

/** Writes the clip space position, a vec4f (y up, depth 0 to w). Vertex stage only. */
void Position(const Val& value, Loc loc = Loc::current());
/** Writes the fragment depth (f32). Fragment stage only. */
void FragDepth(const ValueLike& value, Loc loc = Loc::current());
/**
 * Writes the clip distance at `index` (0 to 7) as an f32. A negative value
 * clips. Vertex stage only. WebGPU needs clip-distances; WebGL2 needs
 * WEBGL_clip_cull_distance.
 */
void ClipDistance(int index, const ValueLike& value, Loc loc = Loc::current());

} // namespace moga
