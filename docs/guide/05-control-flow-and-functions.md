# 5. Control flow and functions

DSL control flow takes C++ lambdas. Each lambda runs one time, when the
program is built. The statements that it records run on the GPU.

## If

```cpp
If(ndotl > 0.0, [&] {
  color.addAssign(diffuse * ndotl);
}).ElseIf(backLit, [&] {
  color.addAssign(diffuse * 0.2);
}).Else([&] {
  color.set(ambient);
});
```

The condition is a `bool` value.

## select

`select(cond, t, f)` returns `t` when `cond` is true, else `f`. A bool
vector selects each component.

The two values are always evaluated. Use `If` to run only one of two
computations.

## Loops

### For

```cpp
For(0, 4, [&](const Val& i) { sum.addAssign(weights[i]); });
For(0, n, [&](const Val& i) { /* ... */ }, 2);        // step 2
For(toU32(0), count, [&](const Val& i) { /* ... */ }); // u32 counter
```

- The loop runs while `i < end`. After each iteration, it adds `step` to
  `i`.
- The counter type is the type of `start`: `i32` (the default) or `u32`.
- A constant loop whose counter wraps around is an error.

For a different condition, or to count down, use `ForRange`:

```cpp
For({.start = 7, .end = 0, .step = -1, .condition = BinaryOp::Ge},
    [&](const Val& i) { /* ... */ });
```

### ForGrid

```cpp
ForGrid(3, 3, [&](const Val& i, const Val& j) { /* ... */ });
```

This runs the body for each `(i, j)` with `0 <= i < rows` and
`0 <= j < cols`.

### While and Loop

```cpp
While(error > 0.001, [&] { /* ... */ });
Loop([&] {
  // ...
  If(done, [] { Break(); });
});
```

`Loop` runs again and again. It must contain `Break()` or `Return()`. A
loop with no exit is an error.

### Break and Continue

- `Break()` stops the loop, or leaves the `Switch` case.
- `Continue()` starts the next iteration.

## Switch

```cpp
Switch(mode)
    .Case(0, [&] { color.set(albedo); })
    .Case({1, 2}, [&] { color.set(normalColor); })
    .Default([&] { color.set(vec3(1.0, 0.0, 1.0)); });
```

- The selector is an `i32` or `u32` value.
- Case values are number literals. A value used two times is an error.
- A case does not fall through to the next case.

## Return and Discard

- `Return()` leaves the function or the stage.
- `Return(value)` returns a value from a function.
- `Discard()` discards the fragment. Use it in the fragment stage, or in a
  function that the fragment stage calls.

## Uniform control flow

Some operations need all invocations to reach them together:

- the derivatives `dpdx()`, `dpdy()` and `fwidth()`
- the barriers (compute only)

moga checks this as WGSL does. When a derivative is in a branch that
depends on a non uniform value (for example a varying or a texture value),
moga reports an error with code `Uniformity`.

To correct this, calculate the derivative before the branch:

```cpp
const Val dx = Let(dpdx(uv), "dx");
If(alpha < 0.5, [&] { /* use dx here */ });
```

## Functions

`fn()` defines a shader function:

```cpp
const Function blinn = fn(
    "blinn",
    {{"n", vec3f()}, {"l", vec3f()}, {"v", vec3f()}, {"p", f32()}},
    f32(),
    [](const Args& a) {
      const Val h = Let(normalize(a["l"] + a["v"]), "h");
      Return(pow(max(dot(a["n"], h), 0.0), a["p"]));
    });

// in a stage
const Val spec = blinn(n, l, v, 32.0);
```

- `params` gives the name and the type of each parameter, in order.
- Read a parameter by name (`a["n"]`) or by position (`a[0]`).
- Use `voidType()` for a function that returns no value. A call to a void
  function is also a statement.
- The body runs one time, when you call `fn()`.
- A function with a return type must return on each path.
- A function cannot use attributes, varyings, outputs or builtin inputs.
  Give their values as parameters.
- A function cannot call itself.
- Each stage gets only the functions that it calls.

Declare functions in a `ShaderBuild`. Then errors in their bodies become
diagnostics of the build. See [2. Concepts](02-concepts.md).
