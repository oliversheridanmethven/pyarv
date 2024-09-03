# CMake generated Testfile for 
# Source directory: /Users/oliver/ClionProjects/testing/src/rubik/shapes/tests
# Build directory: /Users/oliver/ClionProjects/testing/build/src/rubik/shapes/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[shapes]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/rubik/shapes/tests/shapes.py")
set_tests_properties([=[shapes]=] PROPERTIES  ENVIRONMENT "PYTHONPATH=:" _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;165;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;158;add_python_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;153;add_python_tests;/Users/oliver/ClionProjects/testing/src/rubik/shapes/tests/CMakeLists.txt;1;add_all_python_tests;/Users/oliver/ClionProjects/testing/src/rubik/shapes/tests/CMakeLists.txt;0;")
add_test([=[shapes_cpp]=] "/Users/oliver/ClionProjects/testing/build/src/rubik/shapes/tests/shapes_cpp")
set_tests_properties([=[shapes_cpp]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/src/rubik/shapes/tests/CMakeLists.txt;5;add_test;/Users/oliver/ClionProjects/testing/src/rubik/shapes/tests/CMakeLists.txt;0;")
