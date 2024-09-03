add_executable(example example.c)
target_link_libraries(example testing)
add_test(NAME example COMMAND example)

add_python_tests(basic_python.py basic_python_again.py)
