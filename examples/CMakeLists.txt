# The examples are a tutorial: read them in number order. Each NN_name.cpp
# builds the target example_name.
file(GLOB example_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[0-9][0-9]_*.cpp)
foreach(source ${example_sources})
  get_filename_component(file ${source} NAME_WE)
  string(REGEX REPLACE "^[0-9]+_" "" example ${file})
  add_executable(example_${example} ${source})
  target_link_libraries(example_${example} PRIVATE moga::moga)
endforeach()
# the benchmark times each build step, which only the internal header exposes
target_include_directories(example_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
