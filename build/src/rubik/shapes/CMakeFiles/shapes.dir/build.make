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
include src/rubik/shapes/CMakeFiles/shapes.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/rubik/shapes/CMakeFiles/shapes.dir/compiler_depend.make

# Include the progress variables for this target.
include src/rubik/shapes/CMakeFiles/shapes.dir/progress.make

# Include the compile flags for this target's objects.
include src/rubik/shapes/CMakeFiles/shapes.dir/flags.make

src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.o: src/rubik/shapes/CMakeFiles/shapes.dir/flags.make
src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.o: /Users/oliver/ClionProjects/testing/src/rubik/shapes/shape.cpp
src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.o: src/rubik/shapes/CMakeFiles/shapes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.o"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && /usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.o -MF CMakeFiles/shapes.dir/shape.cpp.o.d -o CMakeFiles/shapes.dir/shape.cpp.o -c /Users/oliver/ClionProjects/testing/src/rubik/shapes/shape.cpp

src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/shapes.dir/shape.cpp.i"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && /usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/oliver/ClionProjects/testing/src/rubik/shapes/shape.cpp > CMakeFiles/shapes.dir/shape.cpp.i

src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/shapes.dir/shape.cpp.s"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && /usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/oliver/ClionProjects/testing/src/rubik/shapes/shape.cpp -o CMakeFiles/shapes.dir/shape.cpp.s

src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.o: src/rubik/shapes/CMakeFiles/shapes.dir/flags.make
src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.o: /Users/oliver/ClionProjects/testing/src/rubik/shapes/volume.cpp
src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.o: src/rubik/shapes/CMakeFiles/shapes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.o"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && /usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.o -MF CMakeFiles/shapes.dir/volume.cpp.o.d -o CMakeFiles/shapes.dir/volume.cpp.o -c /Users/oliver/ClionProjects/testing/src/rubik/shapes/volume.cpp

src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/shapes.dir/volume.cpp.i"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && /usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/oliver/ClionProjects/testing/src/rubik/shapes/volume.cpp > CMakeFiles/shapes.dir/volume.cpp.i

src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/shapes.dir/volume.cpp.s"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && /usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/oliver/ClionProjects/testing/src/rubik/shapes/volume.cpp -o CMakeFiles/shapes.dir/volume.cpp.s

# Object files for target shapes
shapes_OBJECTS = \
"CMakeFiles/shapes.dir/shape.cpp.o" \
"CMakeFiles/shapes.dir/volume.cpp.o"

# External object files for target shapes
shapes_EXTERNAL_OBJECTS =

src/rubik/shapes/libshapes.a: src/rubik/shapes/CMakeFiles/shapes.dir/shape.cpp.o
src/rubik/shapes/libshapes.a: src/rubik/shapes/CMakeFiles/shapes.dir/volume.cpp.o
src/rubik/shapes/libshapes.a: src/rubik/shapes/CMakeFiles/shapes.dir/build.make
src/rubik/shapes/libshapes.a: src/rubik/shapes/CMakeFiles/shapes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libshapes.a"
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && $(CMAKE_COMMAND) -P CMakeFiles/shapes.dir/cmake_clean_target.cmake
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/shapes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/rubik/shapes/CMakeFiles/shapes.dir/build: src/rubik/shapes/libshapes.a
.PHONY : src/rubik/shapes/CMakeFiles/shapes.dir/build

src/rubik/shapes/CMakeFiles/shapes.dir/clean:
	cd /Users/oliver/ClionProjects/testing/build/src/rubik/shapes && $(CMAKE_COMMAND) -P CMakeFiles/shapes.dir/cmake_clean.cmake
.PHONY : src/rubik/shapes/CMakeFiles/shapes.dir/clean

src/rubik/shapes/CMakeFiles/shapes.dir/depend:
	cd /Users/oliver/ClionProjects/testing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/testing /Users/oliver/ClionProjects/testing/src/rubik/shapes /Users/oliver/ClionProjects/testing/build /Users/oliver/ClionProjects/testing/build/src/rubik/shapes /Users/oliver/ClionProjects/testing/build/src/rubik/shapes/CMakeFiles/shapes.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/rubik/shapes/CMakeFiles/shapes.dir/depend

