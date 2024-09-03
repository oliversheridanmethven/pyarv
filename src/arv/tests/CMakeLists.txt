add_executable(example_c example.c)
target_link_libraries(example_c testing)
add_test(NAME example_c COMMAND example_c)
