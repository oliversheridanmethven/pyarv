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
include src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/compiler_depend.make

# Include the progress variables for this target.
include src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/progress.make

# Include the compile flags for this target's objects.
include src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/flags.make

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/flags.make
src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o: /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_examples_bindings.c
src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o -MF CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o.d -o CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o -c /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_examples_bindings.c

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.i"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_examples_bindings.c > CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.i

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.s"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_examples_bindings.c -o CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.s

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/flags.make
src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o: /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_binding_module.c
src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o -MF CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o.d -o CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o -c /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_binding_module.c

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.i"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_binding_module.c > CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.i

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.s"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && /usr/local/bin/gcc-14 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples/numpy_binding_module.c -o CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.s

# Object files for target numpy_examples_bindings
numpy_examples_bindings_OBJECTS = \
"CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o" \
"CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o"

# External object files for target numpy_examples_bindings
numpy_examples_bindings_EXTERNAL_OBJECTS =

src/pyarv/bindings/examples/numpy_examples_bindings.so: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_examples_bindings.c.o
src/pyarv/bindings/examples/numpy_examples_bindings.so: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/numpy_binding_module.c.o
src/pyarv/bindings/examples/numpy_examples_bindings.so: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/build.make
src/pyarv/bindings/examples/numpy_examples_bindings.so: /Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
src/pyarv/bindings/examples/numpy_examples_bindings.so: src/arv/examples/libexamples.a
src/pyarv/bindings/examples/numpy_examples_bindings.so: /Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
src/pyarv/bindings/examples/numpy_examples_bindings.so: src/arv/error_codes/liberror_codes.a
src/pyarv/bindings/examples/numpy_examples_bindings.so: /Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
src/pyarv/bindings/examples/numpy_examples_bindings.so: src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/pyarv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C shared module numpy_examples_bindings.so"
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/numpy_examples_bindings.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/build: src/pyarv/bindings/examples/numpy_examples_bindings.so
.PHONY : src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/build

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/clean:
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples && $(CMAKE_COMMAND) -P CMakeFiles/numpy_examples_bindings.dir/cmake_clean.cmake
.PHONY : src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/clean

src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/depend:
	cd /Users/oliver/ClionProjects/pyarv/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/pyarv/bindings/examples /Users/oliver/ClionProjects/pyarv/cmake-build-debug /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples /Users/oliver/ClionProjects/pyarv/cmake-build-debug/src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/pyarv/bindings/examples/CMakeFiles/numpy_examples_bindings.dir/depend

