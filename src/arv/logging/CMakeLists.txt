add_subdirectory(tests)

add_library(logging INTERFACE)
target_include_directories(logging INTERFACE .)
