# CMake generated Testfile for 
# Source directory: /Users/oliver/ClionProjects/testing/src/testing/test
# Build directory: /Users/oliver/ClionProjects/testing/build/src/testing/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[example]=] "/Users/oliver/ClionProjects/testing/build/src/testing/test/example")
set_tests_properties([=[example]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/src/testing/test/CMakeLists.txt;3;add_test;/Users/oliver/ClionProjects/testing/src/testing/test/CMakeLists.txt;0;")
add_test([=[basic_python]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/testing/test/basic_python.py")
set_tests_properties([=[basic_python]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;63;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;55;add_python_test;/Users/oliver/ClionProjects/testing/src/testing/test/CMakeLists.txt;5;add_python_tests;/Users/oliver/ClionProjects/testing/src/testing/test/CMakeLists.txt;0;")
add_test([=[basic_python_again]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/testing/test/basic_python_again.py")
set_tests_properties([=[basic_python_again]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;63;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;55;add_python_test;/Users/oliver/ClionProjects/testing/src/testing/test/CMakeLists.txt;5;add_python_tests;/Users/oliver/ClionProjects/testing/src/testing/test/CMakeLists.txt;0;")
