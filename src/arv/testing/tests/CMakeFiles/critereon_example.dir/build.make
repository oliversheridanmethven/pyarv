# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/x64/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/oliver/ClionProjects/pyarv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/oliver/ClionProjects/pyarv

# Include any dependencies generated for this target.
include src/arv/testing/tests/CMakeFiles/critereon_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/arv/testing/tests/CMakeFiles/critereon_example.dir/compiler_depend.make

# Include the progress variables for this target.
include src/arv/testing/tests/CMakeFiles/critereon_example.dir/progress.make

# Include the compile flags for this target's objects.
include src/arv/testing/tests/CMakeFiles/critereon_example.dir/flags.make

src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.o: src/arv/testing/tests/CMakeFiles/critereon_example.dir/flags.make
src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.o: src/arv/testing/tests/criterion_example.c
src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.o: src/arv/testing/tests/CMakeFiles/critereon_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.o"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.o -MF CMakeFiles/critereon_example.dir/criterion_example.c.o.d -o CMakeFiles/critereon_example.dir/criterion_example.c.o -c /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests/criterion_example.c

src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/critereon_example.dir/criterion_example.c.i"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests/criterion_example.c > CMakeFiles/critereon_example.dir/criterion_example.c.i

src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/critereon_example.dir/criterion_example.c.s"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests/criterion_example.c -o CMakeFiles/critereon_example.dir/criterion_example.c.s

# Object files for target critereon_example
critereon_example_OBJECTS = \
"CMakeFiles/critereon_example.dir/criterion_example.c.o"

# External object files for target critereon_example
critereon_example_EXTERNAL_OBJECTS =

bin/critereon_example: src/arv/testing/tests/CMakeFiles/critereon_example.dir/criterion_example.c.o
bin/critereon_example: src/arv/testing/tests/CMakeFiles/critereon_example.dir/build.make
bin/critereon_example: /Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
bin/critereon_example: src/arv/testing/tests/CMakeFiles/critereon_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/pyarv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../../../bin/critereon_example"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/critereon_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/arv/testing/tests/CMakeFiles/critereon_example.dir/build: bin/critereon_example
.PHONY : src/arv/testing/tests/CMakeFiles/critereon_example.dir/build

src/arv/testing/tests/CMakeFiles/critereon_example.dir/clean:
	cd /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests && $(CMAKE_COMMAND) -P CMakeFiles/critereon_example.dir/cmake_clean.cmake
.PHONY : src/arv/testing/tests/CMakeFiles/critereon_example.dir/clean

src/arv/testing/tests/CMakeFiles/critereon_example.dir/depend:
	cd /Users/oliver/ClionProjects/pyarv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests /Users/oliver/ClionProjects/pyarv/src/arv/testing/tests/CMakeFiles/critereon_example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/arv/testing/tests/CMakeFiles/critereon_example.dir/depend

