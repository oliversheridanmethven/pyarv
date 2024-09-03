# CMake generated Testfile for 
# Source directory: /Users/oliver/ClionProjects/testing/src/media/tests
# Build directory: /Users/oliver/ClionProjects/testing/build/src/media/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[images]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/media/tests/images.py")
set_tests_properties([=[images]=] PROPERTIES  ENVIRONMENT "PYTHONPATH=:" _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;156;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;149;add_python_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;144;add_python_tests;/Users/oliver/ClionProjects/testing/src/media/tests/CMakeLists.txt;1;add_all_python_tests;/Users/oliver/ClionProjects/testing/src/media/tests/CMakeLists.txt;0;")
add_test([=[window]=] "/Users/oliver/ClionProjects/testing/venv/bin/python3.11" "/Users/oliver/ClionProjects/testing/src/media/tests/window.py")
set_tests_properties([=[window]=] PROPERTIES  ENVIRONMENT "PYTHONPATH=:" PROCESSORS "8" PROCESSOR_AFFINITY "ON" _BACKTRACE_TRIPLES "/Users/oliver/ClionProjects/testing/CMakeLists.txt;156;add_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;149;add_python_test;/Users/oliver/ClionProjects/testing/CMakeLists.txt;144;add_python_tests;/Users/oliver/ClionProjects/testing/src/media/tests/CMakeLists.txt;1;add_all_python_tests;/Users/oliver/ClionProjects/testing/src/media/tests/CMakeLists.txt;0;")
