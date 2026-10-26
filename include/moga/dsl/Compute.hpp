#pragma once

// WebGPU only features: compute programs, storage buffers and textures,
// workgroup arrays, barriers, atomics and subgroups. The GLSL backend rejects
// a program that uses any of them.

#include "moga/dsl/Value.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace moga {

// ------------------------------------------------------------------ storage

/**
 * A runtime sized array in the storage address space. Read an element with
 * `buffer[i]`; assign to it when the access is ReadWrite. Read/write buffers
 * are not valid in the vertex stage.
 */
class StorageBuffer {
public:
  /** Wraps a storage buffer resource. Use storage() to declare one. */
  explicit StorageBuffer(StorageBufferPtr resource) : resource_(std::move(resource)) {}
  /** Returns the resource. */
  const StorageBufferPtr& resource() const { return resource_; }
  /** Returns the element at a u32 or i32 index. A constant negative index is an error. */
  Val operator[](const ValueLike& index) const;
  /** Returns the number of elements (u32). */
  Val length() const;

private:
  StorageBufferPtr resource_;
};

/**
 * Declares a storage buffer of `elementType` elements. WebGPU only. Use Read
 * or ReadWrite access; Write is an error. An element type with atomics needs
 * ReadWrite.
 */
StorageBuffer storage(std::string name, TypePtr elementType,
                      StorageAccess access = StorageAccess::ReadWrite,
                      ResourceFrequency frequency = ResourceFrequency::Object, Loc loc = Loc::current());

/** A texture that the shader writes or reads by texel, without a sampler. */
class StorageTexture {
public:
  /** Wraps a storage texture resource. Use storageTexture() to declare one. */
  explicit StorageTexture(StorageTexturePtr resource) : resource_(std::move(resource)) {}
  /** Returns the resource. */
  const StorageTexturePtr& resource() const { return resource_; }
  /**
   * Writes one texel at integer `coord` (vec2i or vec2u, vec3i or vec3u for 3D). Give
   * `layer` for a 2D array only. Not valid for Read access or in the vertex stage.
   */
  void store(const Val& coord, const ValueLike& value,
             const std::optional<ValueLike>& layer = std::nullopt, Loc loc = Loc::current()) const;
  /** Reads one texel at integer `coord`. Give `layer` for a 2D array only. Not valid for Write access. */
  Val load(const Val& coord, const std::optional<ValueLike>& layer = std::nullopt) const;
  /** Returns the size in texels: vec2u for 2D, vec3u for 3D and arrays (layers last). */
  Val size() const;

private:
  StorageTexturePtr resource_;
};

/**
 * Declares a storage texture (2D, 3D or 2D array). WebGPU only. A cube
 * texture is an error. Some formats and accesses need a device feature; see
 * storageFormatFeatures().
 */
StorageTexture storageTexture(std::string name, TextureDimension dimension, TexelFormat format,
                              StorageAccess access = StorageAccess::Write,
                              ResourceFrequency frequency = ResourceFrequency::Material, Loc loc = Loc::current());

/** Declares an array shared by the invocations of one workgroup. Compute stage only. */
Val workgroupArray(std::string name, TypePtr elementType, int length, Loc loc = Loc::current());

// ------------------------------------------------------------------ builtins

/** Returns the global invocation id (vec3u). Compute stage only. */
Val globalId();
/** Returns the invocation id in its workgroup (vec3u). Compute stage only. */
Val localId();
/** Returns the workgroup id (vec3u). Compute stage only. */
Val workgroupId();
/** Returns the number of workgroups of the dispatch (vec3u). Compute stage only. */
Val numWorkgroups();
/** Returns the linear invocation index in its workgroup (u32). Compute stage only. */
Val localIndex();
/** Returns the subgroup size (u32). Compute and fragment stages; needs the subgroups feature. */
Val subgroupSize();
/** Returns the invocation id in its subgroup (u32). Needs the subgroups feature. */
Val subgroupInvocationId();

// ------------------------------------------------------------------ synchronization

/** Waits for all invocations of the workgroup; makes workgroup memory visible. Compute only. */
void workgroupBarrier(Loc loc = Loc::current());
/** Waits for all invocations of the workgroup; makes storage memory visible. Compute only. */
void storageBarrier(Loc loc = Loc::current());
/** Waits for all invocations of the workgroup; makes storage texture writes visible. Compute only. */
void textureBarrier(Loc loc = Loc::current());

// ------------------------------------------------------------------ atomics

// The target is an atomic element of a read/write storage buffer or of a
// workgroup array. Each read modify write returns the old value in a new
// constant, so the result can be used or ignored.
/** Reads an atomic value. The target is an atomic in a ReadWrite storage buffer or a workgroup array. */
Val atomicLoad(const Val& target);
/** Writes an atomic value. Not valid in the vertex stage. */
void atomicStore(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Adds `value` to the atomic and returns the old value. Not valid in the vertex stage. */
Val atomicAdd(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Subtracts `value` from the atomic and returns the old value. Not valid in the vertex stage. */
Val atomicSub(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Sets the atomic to the maximum of it and `value`, and returns the old value. Not valid in the vertex stage. */
Val atomicMax(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Sets the atomic to the minimum of it and `value`, and returns the old value. Not valid in the vertex stage. */
Val atomicMin(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Applies a bitwise AND with `value` to the atomic and returns the old value. Not valid in the vertex stage. */
Val atomicAnd(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Applies a bitwise OR with `value` to the atomic and returns the old value. Not valid in the vertex stage. */
Val atomicOr(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Applies a bitwise XOR with `value` to the atomic and returns the old value. Not valid in the vertex stage. */
Val atomicXor(const Val& target, const ValueLike& value, Loc loc = Loc::current());
/** Sets the atomic to `value` and returns the old value. Not valid in the vertex stage. */
Val atomicExchange(const Val& target, const ValueLike& value, Loc loc = Loc::current());

// ------------------------------------------------------------------ subgroups

// Subgroup operations. They need the subgroups feature and are valid in the
// compute and fragment stages. The reductions take a scalar or vector.
/** Returns the sum of a numeric value over the active invocations. Needs subgroups; compute and fragment stages. */
Val subgroupAdd(const Val& v);
/** Returns the product of a numeric value over the active invocations. Needs subgroups. */
Val subgroupMul(const Val& v);
/** Returns the minimum of a numeric value over the active invocations. Needs subgroups. */
Val subgroupMin(const Val& v);
/** Returns the maximum of a numeric value over the active invocations. Needs subgroups. */
Val subgroupMax(const Val& v);
/** Returns the bitwise AND of an integer over the active invocations. Needs subgroups. */
Val subgroupAnd(const Val& v);
/** Returns the bitwise OR of an integer over the active invocations. Needs subgroups. */
Val subgroupOr(const Val& v);
/** Returns the bitwise XOR of an integer over the active invocations. Needs subgroups. */
Val subgroupXor(const Val& v);
/** Returns true when the bool is true in all active invocations. */
Val subgroupAll(const Val& v);
/** Returns true when the bool is true in any active invocation. */
Val subgroupAny(const Val& v);
/** Returns the value of the first active invocation. */
Val subgroupBroadcastFirst(const Val& v);
/** Returns the value of the invocation `id`. The id must be a constant u32 in 0..127. */
Val subgroupBroadcast(const Val& v, const ValueLike& id);
/** Returns the value of the invocation `id`. The id can change per invocation. */
Val subgroupShuffle(const Val& v, const ValueLike& id);
/** Returns true in exactly one active invocation. */
Val subgroupElect();
/** Returns a bit mask (vec4u) of the invocations where the bool is true. */
Val subgroupBallot(const Val& v);

// ------------------------------------------------------------------ program

/** Options of a compute program. */
struct ComputeOptions {
  /** The workgroup size (@workgroup_size). Each size must be positive. */
  std::array<std::uint32_t, 3> workgroupSize{1, 1, 1};
  /**
   * Runs the program with subgroups of this size, a power of 2 that divides
   * workgroupSize[0] (@subgroup_size). The device needs subgroup-size-control,
   * and the size must be in the subgroup size range of the adapter.
   */
  std::optional<std::uint32_t> subgroupSize;
};

} // namespace moga
