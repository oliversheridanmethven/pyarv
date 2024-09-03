# CMake generated Testfile for 
# Source directory: /Users/oliver/ClionProjects/testing/src/rubiks/shufflers/tests
# Build directory: /Users/oliver/ClionProjects/testing/build/src/rubiks/shufflers/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[shufflers]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/rubiks/shufflers/tests/shufflers.py")
set_tests_properties([=[shufflers]=] PROPERTIES  ENVIRONMENT "PYTHONPATH=:" _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;156;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;149;add_python_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;144;add_python_tests;/Users/oliver/ClionProjects/testing/src/rubiks/shufflers/tests/CMakeLists.txt;1;add_all_python_tests;/Users/oliver/ClionProjects/testing/src/rubiks/shufflers/tests/CMakeLists.txt;0;")
add_test([=[shufflers_cpp]=] "/Users/oliver/ClionProjects/testing/build/src/rubiks/shufflers/tests/shufflers_cpp")
set_tests_properties([=[shufflers_cpp]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/src/rubiks/shufflers/tests/CMakeLists.txt;5;add_test;/Users/oliver/ClionProjects/testing/src/rubiks/shufflers/tests/CMakeLists.txt;0;")
