add_executable(logging_test logging.c)
target_link_libraries(logging_test testing logging)
add_test(NAME logging_test COMMAND logging_test)
