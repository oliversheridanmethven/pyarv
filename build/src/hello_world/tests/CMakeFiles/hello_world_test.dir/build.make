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
include src/hello_world/tests/CMakeFiles/hello_world_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/hello_world/tests/CMakeFiles/hello_world_test.dir/compiler_depend.make

# Include the progress variables for this target.
include src/hello_world/tests/CMakeFiles/hello_world_test.dir/progress.make

# Include the compile flags for this target's objects.
include src/hello_world/tests/CMakeFiles/hello_world_test.dir/flags.make

src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.o: src/hello_world/tests/CMakeFiles/hello_world_test.dir/flags.make
src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.o: /Users/oliver/ClionProjects/testing/src/hello_world/tests/hello_world.c
src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.o: src/hello_world/tests/CMakeFiles/hello_world_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.o"
	cd /Users/oliver/ClionProjects/testing/build/src/hello_world/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.o -MF CMakeFiles/hello_world_test.dir/hello_world.c.o.d -o CMakeFiles/hello_world_test.dir/hello_world.c.o -c /Users/oliver/ClionProjects/testing/src/hello_world/tests/hello_world.c

src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/hello_world_test.dir/hello_world.c.i"
	cd /Users/oliver/ClionProjects/testing/build/src/hello_world/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/testing/src/hello_world/tests/hello_world.c > CMakeFiles/hello_world_test.dir/hello_world.c.i

src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/hello_world_test.dir/hello_world.c.s"
	cd /Users/oliver/ClionProjects/testing/build/src/hello_world/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/testing/src/hello_world/tests/hello_world.c -o CMakeFiles/hello_world_test.dir/hello_world.c.s

# Object files for target hello_world_test
hello_world_test_OBJECTS = \
"CMakeFiles/hello_world_test.dir/hello_world.c.o"

# External object files for target hello_world_test
hello_world_test_EXTERNAL_OBJECTS =

src/hello_world/tests/hello_world_test: src/hello_world/tests/CMakeFiles/hello_world_test.dir/hello_world.c.o
src/hello_world/tests/hello_world_test: src/hello_world/tests/CMakeFiles/hello_world_test.dir/build.make
src/hello_world/tests/hello_world_test: /Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
src/hello_world/tests/hello_world_test: src/hello_world/libhello_world_c.a
src/hello_world/tests/hello_world_test: /usr/local/lib/libgtest_main.a
src/hello_world/tests/hello_world_test: /usr/local/lib/libgtest.a
src/hello_world/tests/hello_world_test: /Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
src/hello_world/tests/hello_world_test: src/hello_world/tests/CMakeFiles/hello_world_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hello_world_test"
	cd /Users/oliver/ClionProjects/testing/build/src/hello_world/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_world_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/hello_world/tests/CMakeFiles/hello_world_test.dir/build: src/hello_world/tests/hello_world_test
.PHONY : src/hello_world/tests/CMakeFiles/hello_world_test.dir/build

src/hello_world/tests/CMakeFiles/hello_world_test.dir/clean:
	cd /Users/oliver/ClionProjects/testing/build/src/hello_world/tests && $(CMAKE_COMMAND) -P CMakeFiles/hello_world_test.dir/cmake_clean.cmake
.PHONY : src/hello_world/tests/CMakeFiles/hello_world_test.dir/clean

src/hello_world/tests/CMakeFiles/hello_world_test.dir/depend:
	cd /Users/oliver/ClionProjects/testing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/testing /Users/oliver/ClionProjects/testing/src/hello_world/tests /Users/oliver/ClionProjects/testing/build /Users/oliver/ClionProjects/testing/build/src/hello_world/tests /Users/oliver/ClionProjects/testing/build/src/hello_world/tests/CMakeFiles/hello_world_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/hello_world/tests/CMakeFiles/hello_world_test.dir/depend

