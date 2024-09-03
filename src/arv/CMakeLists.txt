add_subdirectory(tests)
add_subdirectory(version)
add_subdirectory(testing)
add_subdirectory(logging)
add_subdirectory(error_codes)

add_library(arv INTERFACE)
target_include_directories(arv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
