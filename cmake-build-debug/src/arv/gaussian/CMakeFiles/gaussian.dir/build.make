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
include src/arv/gaussian/CMakeFiles/gaussian.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/arv/gaussian/CMakeFiles/gaussian.dir/compiler_depend.make

# Include the progress variables for this target.
include src/arv/gaussian/CMakeFiles/gaussian.dir/progress.make

# Include the compile flags for this target's objects.
include src/arv/gaussian/CMakeFiles/gaussian.dir/flags.make

src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.o: src/arv/gaussian/CMakeFiles/gaussian.dir/flags.make
src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.o: /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/polynomial.c
src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.o: src/arv/gaussian/CMakeFiles/gaussian.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.o"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.o -MF CMakeFiles/gaussian.dir/polynomial.c.o.d -o CMakeFiles/gaussian.dir/polynomial.c.o -c /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/polynomial.c

src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/gaussian.dir/polynomial.c.i"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/polynomial.c > CMakeFiles/gaussian.dir/polynomial.c.i

src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/gaussian.dir/polynomial.c.s"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/polynomial.c -o CMakeFiles/gaussian.dir/polynomial.c.s

src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.o: src/arv/gaussian/CMakeFiles/gaussian.dir/flags.make
src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.o: /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/linear.c
src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.o: src/arv/gaussian/CMakeFiles/gaussian.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.o"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.o -MF CMakeFiles/gaussian.dir/linear.c.o.d -o CMakeFiles/gaussian.dir/linear.c.o -c /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/linear.c

src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/gaussian.dir/linear.c.i"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/linear.c > CMakeFiles/gaussian.dir/linear.c.i

src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/gaussian.dir/linear.c.s"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/linear.c -o CMakeFiles/gaussian.dir/linear.c.s

src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.o: src/arv/gaussian/CMakeFiles/gaussian.dir/flags.make
src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.o: /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/cubic.c
src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.o: src/arv/gaussian/CMakeFiles/gaussian.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.o"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.o -MF CMakeFiles/gaussian.dir/cubic.c.o.d -o CMakeFiles/gaussian.dir/cubic.c.o -c /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/cubic.c

src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/gaussian.dir/cubic.c.i"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/cubic.c > CMakeFiles/gaussian.dir/cubic.c.i

src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/gaussian.dir/cubic.c.s"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/arv/gaussian/cubic.c -o CMakeFiles/gaussian.dir/cubic.c.s

# Object files for target gaussian
gaussian_OBJECTS = \
"CMakeFiles/gaussian.dir/polynomial.c.o" \
"CMakeFiles/gaussian.dir/linear.c.o" \
"CMakeFiles/gaussian.dir/cubic.c.o"

# External object files for target gaussian
gaussian_EXTERNAL_OBJECTS =

src/arv/gaussian/libgaussian.a: src/arv/gaussian/CMakeFiles/gaussian.dir/polynomial.c.o
src/arv/gaussian/libgaussian.a: src/arv/gaussian/CMakeFiles/gaussian.dir/linear.c.o
src/arv/gaussian/libgaussian.a: src/arv/gaussian/CMakeFiles/gaussian.dir/cubic.c.o
src/arv/gaussian/libgaussian.a: src/arv/gaussian/CMakeFiles/gaussian.dir/build.make
src/arv/gaussian/libgaussian.a: src/arv/gaussian/CMakeFiles/gaussian.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C static library libgaussian.a"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && $(CMAKE_COMMAND) -P CMakeFiles/gaussian.dir/cmake_clean_target.cmake
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gaussian.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/arv/gaussian/CMakeFiles/gaussian.dir/build: src/arv/gaussian/libgaussian.a
.PHONY : src/arv/gaussian/CMakeFiles/gaussian.dir/build

src/arv/gaussian/CMakeFiles/gaussian.dir/clean:
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian && $(CMAKE_COMMAND) -P CMakeFiles/gaussian.dir/cmake_clean.cmake
.PHONY : src/arv/gaussian/CMakeFiles/gaussian.dir/clean

src/arv/gaussian/CMakeFiles/gaussian.dir/depend:
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/arv/gaussian /Users/oliver/ClionProjects/pyarv/cmake-build-debug /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/arv/gaussian/CMakeFiles/gaussian.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/arv/gaussian/CMakeFiles/gaussian.dir/depend

