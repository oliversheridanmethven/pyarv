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
CMAKE_BINARY_DIR = /Users/oliver/ClionProjects/pyarv/cmake-build-debug

# Include any dependencies generated for this target.
include src/arv/error_codes/CMakeFiles/error_codes.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/arv/error_codes/CMakeFiles/error_codes.dir/compiler_depend.make

# Include the progress variables for this target.
include src/arv/error_codes/CMakeFiles/error_codes.dir/progress.make

# Include the compile flags for this target's objects.
include src/arv/error_codes/CMakeFiles/error_codes.dir/flags.make

src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.o: src/arv/error_codes/CMakeFiles/error_codes.dir/flags.make
src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.o: /Users/oliver/ClionProjects/pyarv/src/arv/error_codes/error_codes.c
src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.o: src/arv/error_codes/CMakeFiles/error_codes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.o"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.o -MF CMakeFiles/error_codes.dir/error_codes.c.o.d -o CMakeFiles/error_codes.dir/error_codes.c.o -c /Users/oliver/ClionProjects/pyarv/src/arv/error_codes/error_codes.c

src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/error_codes.dir/error_codes.c.i"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/arv/error_codes/error_codes.c > CMakeFiles/error_codes.dir/error_codes.c.i

src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/error_codes.dir/error_codes.c.s"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/arv/error_codes/error_codes.c -o CMakeFiles/error_codes.dir/error_codes.c.s

# Object files for target error_codes
error_codes_OBJECTS = \
"CMakeFiles/error_codes.dir/error_codes.c.o"

# External object files for target error_codes
error_codes_EXTERNAL_OBJECTS =

src/arv/error_codes/liberror_codes.a: src/arv/error_codes/CMakeFiles/error_codes.dir/error_codes.c.o
src/arv/error_codes/liberror_codes.a: src/arv/error_codes/CMakeFiles/error_codes.dir/build.make
src/arv/error_codes/liberror_codes.a: src/arv/error_codes/CMakeFiles/error_codes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library liberror_codes.a"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes && $(CMAKE_COMMAND) -P CMakeFiles/error_codes.dir/cmake_clean_target.cmake
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/error_codes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/arv/error_codes/CMakeFiles/error_codes.dir/build: src/arv/error_codes/liberror_codes.a
.PHONY : src/arv/error_codes/CMakeFiles/error_codes.dir/build

src/arv/error_codes/CMakeFiles/error_codes.dir/clean:
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes && $(CMAKE_COMMAND) -P CMakeFiles/error_codes.dir/cmake_clean.cmake
.PHONY : src/arv/error_codes/CMakeFiles/error_codes.dir/clean

src/arv/error_codes/CMakeFiles/error_codes.dir/depend:
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/arv/error_codes /Users/oliver/ClionProjects/pyarv/cmake-build-debug /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/error_codes/CMakeFiles/error_codes.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/arv/error_codes/CMakeFiles/error_codes.dir/depend

