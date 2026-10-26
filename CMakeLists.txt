cmake_minimum_required(VERSION 3.25)
project(moga VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MOGA_BUILD_EXAMPLES "Build the examples" ON)
option(MOGA_BUILD_TESTS "Build the tests" ON)

add_library(moga
  src/Analyze.cpp
  src/Build.cpp
  src/Capabilities.cpp
  src/Half.cpp
  src/Known.cpp
  src/Lower.cpp
  src/Scopes.cpp
  src/Uniformity.cpp
  src/Walk.cpp
  src/ControlFlow.cpp
  src/BuiltinTable.cpp
  src/dsl/Builtins.cpp
  src/dsl/Compute.cpp
  src/dsl/Constants.cpp
  src/dsl/Constructors.cpp
  src/dsl/Context.cpp
  src/dsl/Declarations.cpp
  src/dsl/Functions.cpp
  src/dsl/Math.cpp
  src/dsl/Program.cpp
  src/dsl/Statements.cpp
  src/dsl/Value.cpp
  src/GLSL.cpp
  src/Layout.cpp
  src/Nodes.cpp
  src/TextEmitter.cpp
  src/Types.cpp
  src/Variant.cpp
  src/WGSL.cpp)
add_library(moga::moga ALIAS moga)
target_include_directories(moga
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_compile_features(moga PUBLIC cxx_std_23)
if(MSVC)
  target_compile_options(moga PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(moga PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)
install(TARGETS moga EXPORT mogaTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mogaTargets NAMESPACE moga:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/moga)

# find_package(moga CONFIG) needs a config file next to the exported targets
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/mogaConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/mogaConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/moga)
# before 1.0 a minor version may break the API
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mogaConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mogaConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/mogaConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/moga)

if(MOGA_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MOGA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
