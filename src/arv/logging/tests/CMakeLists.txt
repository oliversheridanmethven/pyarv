add_executable(logging_test test_logging.c)
target_link_libraries(logging_test testing logging)
add_test(NAME test_logging_example COMMAND logging_test)
