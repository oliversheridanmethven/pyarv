add_subdirectory(tests)

add_library(simple_examples_bindings MODULE
        simple_examples_bindings.c simple_binding_module.c
)
add_library(numpy_examples_bindings MODULE
        numpy_examples_bindings.c numpy_binding_module.c
)

target_link_libraries(simple_examples_bindings examples bindings)
target_link_libraries(numpy_examples_bindings examples bindings)
install_python_target(simple_examples_bindings src/pyarv/bindings/examples)
install_python_target(numpy_examples_bindings src/pyarv/bindings/examples)
