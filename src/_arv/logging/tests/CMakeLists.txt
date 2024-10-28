add_executable(test_logging logging.c)
target_link_libraries(test_logging testing logging)
add_test(NAME test_logging COMMAND test_logging)
