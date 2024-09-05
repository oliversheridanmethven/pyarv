add_subdirectory(bindings)
add_subdirectory(version)
add_subdirectory(gaussian)

add_library(pyarv INTERFACE)
target_include_directories(pyarv INTERFACE .)
