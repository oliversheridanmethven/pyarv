add_subdirectory(tests)

add_library(gaussian_bindings MODULE
        gaussian_module.c gaussian_bindings.c
)
target_link_libraries(gaussian_bindings bindings)
install_python_target(gaussian_bindings src/pyarv/gaussian)
