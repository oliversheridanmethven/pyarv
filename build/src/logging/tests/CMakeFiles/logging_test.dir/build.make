# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.29.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.29.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/oliver/ClionProjects/testing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/oliver/ClionProjects/testing/build

# Include any dependencies generated for this target.
include src/logging/tests/CMakeFiles/logging_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/logging/tests/CMakeFiles/logging_test.dir/compiler_depend.make

# Include the progress variables for this target.
include src/logging/tests/CMakeFiles/logging_test.dir/progress.make

# Include the compile flags for this target's objects.
include src/logging/tests/CMakeFiles/logging_test.dir/flags.make

src/logging/tests/CMakeFiles/logging_test.dir/logging.c.o: src/logging/tests/CMakeFiles/logging_test.dir/flags.make
src/logging/tests/CMakeFiles/logging_test.dir/logging.c.o: /Users/oliver/ClionProjects/testing/src/logging/tests/logging.c
src/logging/tests/CMakeFiles/logging_test.dir/logging.c.o: src/logging/tests/CMakeFiles/logging_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/logging/tests/CMakeFiles/logging_test.dir/logging.c.o"
	cd /Users/oliver/ClionProjects/testing/build/src/logging/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/logging/tests/CMakeFiles/logging_test.dir/logging.c.o -MF CMakeFiles/logging_test.dir/logging.c.o.d -o CMakeFiles/logging_test.dir/logging.c.o -c /Users/oliver/ClionProjects/testing/src/logging/tests/logging.c

src/logging/tests/CMakeFiles/logging_test.dir/logging.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/logging_test.dir/logging.c.i"
	cd /Users/oliver/ClionProjects/testing/build/src/logging/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/testing/src/logging/tests/logging.c > CMakeFiles/logging_test.dir/logging.c.i

src/logging/tests/CMakeFiles/logging_test.dir/logging.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/logging_test.dir/logging.c.s"
	cd /Users/oliver/ClionProjects/testing/build/src/logging/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/testing/src/logging/tests/logging.c -o CMakeFiles/logging_test.dir/logging.c.s

# Object files for target logging_test
logging_test_OBJECTS = \
"CMakeFiles/logging_test.dir/logging.c.o"

# External object files for target logging_test
logging_test_EXTERNAL_OBJECTS =

src/logging/tests/logging_test: src/logging/tests/CMakeFiles/logging_test.dir/logging.c.o
src/logging/tests/logging_test: src/logging/tests/CMakeFiles/logging_test.dir/build.make
src/logging/tests/logging_test: /Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
src/logging/tests/logging_test: /usr/local/lib/libgtest_main.a
src/logging/tests/logging_test: /usr/local/lib/libgtest.a
src/logging/tests/logging_test: /usr/local/lib/libglog.0.6.0.dylib
src/logging/tests/logging_test: /usr/local/lib/libgflags.2.2.2.dylib
src/logging/tests/logging_test: src/logging/tests/CMakeFiles/logging_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable logging_test"
	cd /Users/oliver/ClionProjects/testing/build/src/logging/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/logging_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/logging/tests/CMakeFiles/logging_test.dir/build: src/logging/tests/logging_test
.PHONY : src/logging/tests/CMakeFiles/logging_test.dir/build

src/logging/tests/CMakeFiles/logging_test.dir/clean:
	cd /Users/oliver/ClionProjects/testing/build/src/logging/tests && $(CMAKE_COMMAND) -P CMakeFiles/logging_test.dir/cmake_clean.cmake
.PHONY : src/logging/tests/CMakeFiles/logging_test.dir/clean

src/logging/tests/CMakeFiles/logging_test.dir/depend:
	cd /Users/oliver/ClionProjects/testing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/testing /Users/oliver/ClionProjects/testing/src/logging/tests /Users/oliver/ClionProjects/testing/build /Users/oliver/ClionProjects/testing/build/src/logging/tests /Users/oliver/ClionProjects/testing/build/src/logging/tests/CMakeFiles/logging_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/logging/tests/CMakeFiles/logging_test.dir/depend

