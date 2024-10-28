add_subdirectory(tests)

add_library(version_bindings MODULE version_bindings.c version_module.c)
target_link_libraries(version_bindings bindings version)
install_python_target(version_bindings src/pyarv/version)



