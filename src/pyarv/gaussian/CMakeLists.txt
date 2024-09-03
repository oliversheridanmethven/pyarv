add_library(examples ../../arv/examples.c)
target_link_libraries(examples error_codes)
target_include_directories(examples INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(examples_bindings MODULE examples_bindings.c binding_module.c)
target_link_libraries(examples_bindings examples binding)
install_python_target(examples_bindings src/binding/examples)