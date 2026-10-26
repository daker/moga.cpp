# 3. Types and values

## Types

A type is a `TypePtr`. Functions give the common types.

### Scalars

| Function | WGSL | GLSL | Notes |
| --- | --- | --- | --- |
| `f32()` | `f32` | `float` | |
| `i32()` | `i32` | `int` | |
| `u32()` | `u32` | `uint` | |
| `boolType()` | `bool` | `bool` | Not valid in uniform blocks or storage. |
| `f16()` | `f16` | none | WebGPU only. Needs `shader-f16`. |

### Vectors

| Component | 2 | 3 | 4 |
| --- | --- | --- | --- |
| f32 | `vec2f()` | `vec3f()` | `vec4f()` |
| i32 | `vec2i()` | `vec3i()` | `vec4i()` |
| u32 | `vec2u()` | `vec3u()` | `vec4u()` |
| bool | `vec2b()` | `vec3b()` | `vec4b()` |
| f16 | `vec2h()` | `vec3h()` | `vec4h()` |

`vectorType(kind, size)` gives a vector type from a `ScalarKind` and a
size.

### Matrices

Matrices are square and column major.

| Size | f32 | f16 (WebGPU only) |
| --- | --- | --- |
| 2x2 | `mat2f()` | `mat2h()` |
| 3x3 | `mat3f()` | `mat3h()` |
| 4x4 | `mat4f()` | `mat4h()` |

`mat2f()` is not valid in uniform blocks. See
[8. Reflection and binding](08-reflection-and-binding.md).

### Arrays and structs

```cpp
const TypePtr lights = arrayOf(vec4f(), 8);
const TypePtr particle = makeStruct("Particle", {
    {"position", vec3f()},
    {"mass", f32()},
    {"velocity", vec3f()},
});
```

- An array has a fixed length.
- An array of arrays is WebGPU only.
- Structs compare by name and fields. Use `typesEqual()` to compare types.

### Other types

| Function | Use |
| --- | --- |
| `voidType()` | The return type of a function that returns no value. |
| `atomicU32()`, `atomicI32()` | Atomic elements. WebGPU only. See [10](10-webgpu-only.md). |

## Constructors

Constructors make a value of a type from components or other values. One
scalar fills all components.

```cpp
vec3(0.0)                  // (0, 0, 0)
vec4(position, 1.0)        // a vec3f and one component
vec2(uv.y(), uv.x())
ivec2(3, 4)
mat3(model)                // the upper left 3x3 of a mat4x4f
mat4(1.0)                  // the identity matrix
color(1.0, 0.5, 0.0)       // the same as vec3()
```

| Family | Functions |
| --- | --- |
| f32 vectors | `vec2`, `vec3`, `vec4`, `color` |
| i32 vectors | `ivec2`, `ivec3`, `ivec4` |
| u32 vectors | `uvec2`, `uvec3`, `uvec4` |
| bool vectors | `bvec2`, `bvec3`, `bvec4` |
| f32 matrices | `mat2`, `mat3`, `mat4` |
| f16 (WebGPU only) | `hvec2`, `hvec3`, `hvec4`, `hmat2`, `hmat3`, `hmat4` |

For other types, use `construct(type, {values...})`. For a struct, give one
value for each field, in order. For an array, use `array({values...})`.

## Conversions

| Function | Result |
| --- | --- |
| `toF32(a)`, `toI32(a)`, `toU32(a)`, `toBool(a)`, `toF16(a)` | A scalar of that kind. |
| `cast(v, ScalarKind::I32)` | The same size, another component kind. |
| `convert(v, vec3f())` | Another size and kind. A larger vector keeps its first components. A smaller vector gets zero components at the end. |
| `toVec2`, `toVec3`, `toVec4`, `toIVec2` ... `toBVec4` | `convert()` to that vector type. |

To read the bits of a float, use `floatBitsToUint()` and the related
functions in [6. Builtin functions](06-builtin-functions.md).

## Components

```cpp
v.x(); v.y(); v.z(); v.w();           // one component
v.xy(); v.xz(); v.yz(); v.zw(); v.xyz();
v.rgb(); v.a();                       // color names
v.swizzle("zyx");                     // any swizzle
m[2]                                  // matrix column 2
arr[i]                                // array element; i is i32 or u32
s.member("mass")                      // struct field
```

## Operators

| Operators | Operands |
| --- | --- |
| `+ - * /` | Numbers. `*` with a matrix is the linear algebra product. |
| `%` | Integers only. For floats, use `mod()`. |
| `< <= > >=` | Numbers. The result is `bool`, or a bool vector. |
| `& \| ^ ~ << >>` | Integers. A constant shift must be in 0..31. |
| `&& \|\| !` | Bools. |
| unary `-` | Floats, i32 values and matrices. Not u32. |

Important rules:

- **Equality is `eq()` and `ne()`, not `==` and `!=`.** Write
  `a.eq(b)`.
- **`&&` and `||` do not short circuit.** The two operands are always
  evaluated. Use `If` to run only one computation.
- A scalar operand applies to each component of a vector.

## Variables and constants

| Function | WGSL | Use |
| --- | --- | --- |
| `Var(type, "name")` | `var` | A variable, set to zero. |
| `Var(value, "name")` | `var` | A variable with an initial value. |
| `Let(value, "name")` | `let` | A value calculated one time. You cannot write it. |

The name is a hint. moga adds a number to make it unique.

### Assignment

```cpp
const Val color = Var(vec3(0.0), "color");
color.set(base * light);
color.addAssign(ambient);          // +=
color.mulAssign(0.5);              // *=
color.x().set(1.0);                // a swizzle of a variable
const Val i = Var(toI32(0), "i");
i.increment();                     // ++
```

The compound assignments are `addAssign`, `subAssign`, `mulAssign`,
`divAssign`, `remAssign`, `bitAndAssign`, `bitOrAssign`, `bitXorAssign`,
`shlAssign` and `shrAssign`. Each evaluates the target one time.

You can write:

- a `Var`, or a swizzle, element or member of a `Var`
- an element of a `ReadWrite` storage buffer

## Named constants

`constant()` declares a named shader constant. Make its value from
literals, constructors, operators, builtin functions and other constants
only.

```cpp
const Val kGamma = constant("GAMMA", 2.2);
const Val kInvGamma = constant("INV_GAMMA", 1.0 / kGamma);
```

- Declare a constant one time. Many programs can use it.
- A program writes only the constants that it uses, after the constants
  that they use.

moga has these common f32 constants in namespace `consts`:

| Function | Name in the code | Value |
| --- | --- | --- |
| `consts::pi()` | `PI` | π |
| `consts::twoPi()` | `TWO_PI` | 2π |
| `consts::halfPi()` | `HALF_PI` | π/2 |
| `consts::epsilon()` | `EPSILON` | The difference between 1 and the next f32. |
| `consts::infinity()` | `INFINITY` | Positive infinity. |

## Constant checks

moga checks constant expressions when you build the program. For example,
it rejects an integer division by a constant zero, a constant index out of
range, and a constant shift of 32.

WGSL lets a float result round in more than one way. moga checks all the
values that WGSL permits. Thus a message can say that a value "can be"
zero. See [ADR 0006](../adr/0006-constant-values-as-ranges.md).
