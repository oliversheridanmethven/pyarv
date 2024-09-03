add_subdirectory(tests)

add_library(examples_bindings MODULE examples_bindings.c binding_module.c)
target_link_libraries(examples_bindings examples bindings)
install_python_target(examples_bindings src/pyarv/binding/examples)