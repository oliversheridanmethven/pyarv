# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/oliver/ClionProjects/testing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/oliver/ClionProjects/testing/build

# Include any dependencies generated for this target.
include src/binding/CMakeFiles/examples.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/binding/CMakeFiles/examples.dir/compiler_depend.make

# Include the progress variables for this target.
include src/binding/CMakeFiles/examples.dir/progress.make

# Include the compile flags for this target's objects.
include src/binding/CMakeFiles/examples.dir/flags.make

src/binding/CMakeFiles/examples.dir/examples.c.o: src/binding/CMakeFiles/examples.dir/flags.make
src/binding/CMakeFiles/examples.dir/examples.c.o: /Users/oliver/ClionProjects/testing/src/binding/examples.c
src/binding/CMakeFiles/examples.dir/examples.c.o: src/binding/CMakeFiles/examples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/binding/CMakeFiles/examples.dir/examples.c.o"
	cd /Users/oliver/ClionProjects/testing/build/src/binding && gcc-13 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/binding/CMakeFiles/examples.dir/examples.c.o -MF CMakeFiles/examples.dir/examples.c.o.d -o CMakeFiles/examples.dir/examples.c.o -c /Users/oliver/ClionProjects/testing/src/binding/examples.c

src/binding/CMakeFiles/examples.dir/examples.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/examples.dir/examples.c.i"
	cd /Users/oliver/ClionProjects/testing/build/src/binding && gcc-13 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/testing/src/binding/examples.c > CMakeFiles/examples.dir/examples.c.i

src/binding/CMakeFiles/examples.dir/examples.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/examples.dir/examples.c.s"
	cd /Users/oliver/ClionProjects/testing/build/src/binding && gcc-13 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/testing/src/binding/examples.c -o CMakeFiles/examples.dir/examples.c.s

# Object files for target examples
examples_OBJECTS = \
"CMakeFiles/examples.dir/examples.c.o"

# External object files for target examples
examples_EXTERNAL_OBJECTS =

src/binding/libexamples.a: src/binding/CMakeFiles/examples.dir/examples.c.o
src/binding/libexamples.a: src/binding/CMakeFiles/examples.dir/build.make
src/binding/libexamples.a: src/binding/CMakeFiles/examples.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libexamples.a"
	cd /Users/oliver/ClionProjects/testing/build/src/binding && $(CMAKE_COMMAND) -P CMakeFiles/examples.dir/cmake_clean_target.cmake
	cd /Users/oliver/ClionProjects/testing/build/src/binding && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/examples.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/binding/CMakeFiles/examples.dir/build: src/binding/libexamples.a
.PHONY : src/binding/CMakeFiles/examples.dir/build

src/binding/CMakeFiles/examples.dir/clean:
	cd /Users/oliver/ClionProjects/testing/build/src/binding && $(CMAKE_COMMAND) -P CMakeFiles/examples.dir/cmake_clean.cmake
.PHONY : src/binding/CMakeFiles/examples.dir/clean

src/binding/CMakeFiles/examples.dir/depend:
	cd /Users/oliver/ClionProjects/testing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/testing /Users/oliver/ClionProjects/testing/src/binding /Users/oliver/ClionProjects/testing/build /Users/oliver/ClionProjects/testing/build/src/binding /Users/oliver/ClionProjects/testing/build/src/binding/CMakeFiles/examples.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/binding/CMakeFiles/examples.dir/depend
