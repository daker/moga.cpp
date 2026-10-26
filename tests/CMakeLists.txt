find_package(doctest REQUIRED)
add_executable(moga_tests main.cpp types.test.cpp layout.test.cpp dsl.test.cpp compile.test.cpp features.test.cpp compute.test.cpp robustness.test.cpp build.test.cpp)
target_link_libraries(moga_tests PRIVATE moga::moga doctest::doctest)
# the tests reach the build steps one at a time
target_include_directories(moga_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME moga_tests COMMAND moga_tests)
