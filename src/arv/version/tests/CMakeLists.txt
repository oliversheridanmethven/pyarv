add_executable(version_example version.c)
target_link_libraries(version_example version testing logging)
add_test(NAME version_example COMMAND version_example)
