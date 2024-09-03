add_subdirectory(tests)

add_library(arv INTERFACE)
target_include_directories(arv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
