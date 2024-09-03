# Some basic examples.
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(gaussian)

add_library(binding INTERFACE)
target_include_directories(binding INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
