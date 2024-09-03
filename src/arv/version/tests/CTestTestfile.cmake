# CMake generated Testfile for 
# Source directory: /Users/oliver/ClionProjects/testing/src/version/tests
# Build directory: /Users/oliver/ClionProjects/testing/src/version/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[version_test]=] "/Users/oliver/ClionProjects/testing/bin/version_example")
set_tests_properties([=[version_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/src/version/tests/CMakeLists.txt;3;add_test;/Users/oliver/ClionProjects/testing/src/version/tests/CMakeLists.txt;0;")
add_test([=[version_bindings]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/version/tests/version_bindings.py")
set_tests_properties([=[version_bindings]=] PROPERTIES  ENVIRONMENT "PYTHONPATH=/Users/oliver/ClionProjects/testing/src:" _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;156;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;149;add_python_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;144;add_python_tests;/Users/oliver/ClionProjects/testing/src/version/tests/CMakeLists.txt;5;add_all_python_tests;/Users/oliver/ClionProjects/testing/src/version/tests/CMakeLists.txt;0;")
