add_subdirectory(tests)

add_library(logging SHARED logging.c)
target_include_directories(logging INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(logging PUBLIC ${GLIB_LIBRARIES})
target_compile_options(logging PUBLIC -Wno-constant-logical-operand)  # GLib can trigger this error with some compilers.
