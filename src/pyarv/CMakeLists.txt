add_subdirectory(bindings)
add_subdirectory(version)

add_library(pyarv INTERFACE)
target_include_directories(pyarv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
