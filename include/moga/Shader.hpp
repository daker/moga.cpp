#pragma once

// moga: author shaders once in a C++ DSL; compile to GLSL ES 3.00 (WebGL2)
// and WGSL (WebGPU) with reflection.

#include "moga/Analyze.hpp"
#include "moga/Build.hpp"
#include "moga/Compile.hpp"
#include "moga/dsl/Value.hpp"
#include "moga/dsl/Constants.hpp"
#include "moga/dsl/Declarations.hpp"
#include "moga/dsl/Builtins.hpp"
#include "moga/dsl/Statements.hpp"
#include "moga/dsl/Functions.hpp"
#include "moga/dsl/Constructors.hpp"
#include "moga/dsl/Math.hpp"
#include "moga/dsl/Program.hpp"
#include "moga/dsl/Compute.hpp"
#include "moga/Layout.hpp"
#include "moga/Nodes.hpp"
#include "moga/Types.hpp"
#include "moga/Variant.hpp"
