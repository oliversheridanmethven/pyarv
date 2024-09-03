add_executable(version_example version.c)
target_link_libraries(version_example version testing logging)
add_test(NAME version_test COMMAND version_example)

add_all_python_tests()
