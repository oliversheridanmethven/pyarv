add_subdirectory(tests)
add_subdirectory(demo)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0)
find_package(glog REQUIRED)
include_directories(SYSTEM ${glog_INCLUDE_DIRS})
include_directories(SYSTEM ${GLIB_INCLUDE_DIRS})

add_library(logging INTERFACE)
target_include_directories(logging SYSTEM INTERFACE ${GLIB_INCLUDE_DIRS} )
target_include_directories(logging INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(logging INTERFACE ${GLIB_LIBRARIES} glog::glog)
target_compile_options(logging INTERFACE -Wno-constant-logical-operand)  # GLib can trigger this error with some compilers.

