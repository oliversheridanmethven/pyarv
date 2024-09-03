# CMake generated Testfile for 
# Source directory: /Users/oliver/ClionProjects/testing/src/rubiks/solvers/tests
# Build directory: /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[solvers]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/rubiks/solvers/tests/solvers.py")
set_tests_properties([=[solvers]=] PROPERTIES  ENVIRONMENT "PYTHONPATH=:" _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;156;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;149;add_python_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;144;add_python_tests;/Users/oliver/ClionProjects/testing/src/rubiks/solvers/tests/CMakeLists.txt;1;add_all_python_tests;/Users/oliver/ClionProjects/testing/src/rubiks/solvers/tests/CMakeLists.txt;0;")
add_test([=[solvers_cpp]=] "/Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/tests/solvers_cpp")
set_tests_properties([=[solvers_cpp]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/src/rubiks/solvers/tests/CMakeLists.txt;5;add_test;/Users/oliver/ClionProjects/testing/src/rubiks/solvers/tests/CMakeLists.txt;0;")
