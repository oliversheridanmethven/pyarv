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
include src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/flags.make

src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.o: src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/flags.make
src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.o: /Users/oliver/ClionProjects/testing/src/rubiks/solvers/demos/solver.cpp
src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.o: src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.o"
	cd /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos && /opt/local/bin/clang++-mp-16 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.o -MF CMakeFiles/solver_demo_cpp.dir/solver.cpp.o.d -o CMakeFiles/solver_demo_cpp.dir/solver.cpp.o -c /Users/oliver/ClionProjects/testing/src/rubiks/solvers/demos/solver.cpp

src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/solver_demo_cpp.dir/solver.cpp.i"
	cd /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos && /opt/local/bin/clang++-mp-16 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/oliver/ClionProjects/testing/src/rubiks/solvers/demos/solver.cpp > CMakeFiles/solver_demo_cpp.dir/solver.cpp.i

src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/solver_demo_cpp.dir/solver.cpp.s"
	cd /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos && /opt/local/bin/clang++-mp-16 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/oliver/ClionProjects/testing/src/rubiks/solvers/demos/solver.cpp -o CMakeFiles/solver_demo_cpp.dir/solver.cpp.s

# Object files for target solver_demo_cpp
solver_demo_cpp_OBJECTS = \
"CMakeFiles/solver_demo_cpp.dir/solver.cpp.o"

# External object files for target solver_demo_cpp
solver_demo_cpp_EXTERNAL_OBJECTS =

src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/solver.cpp.o
src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/build.make
src/rubiks/solvers/demos/solver_demo_cpp: /Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/colours/libcolours.a
src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/paths/libpaths.a
src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/colours/libcolours.a
src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/paths/libpaths.a
src/rubiks/solvers/demos/solver_demo_cpp: /usr/local/lib/libglog.0.6.0.dylib
src/rubiks/solvers/demos/solver_demo_cpp: /usr/local/lib/libgflags.2.2.2.dylib
src/rubiks/solvers/demos/solver_demo_cpp: /usr/local/lib/libboost_program_options-mt.dylib
src/rubiks/solvers/demos/solver_demo_cpp: /usr/local/lib/libboost_filesystem-mt.dylib
src/rubiks/solvers/demos/solver_demo_cpp: /usr/local/lib/libboost_atomic-mt.dylib
src/rubiks/solvers/demos/solver_demo_cpp: src/version/libversion.a
src/rubiks/solvers/demos/solver_demo_cpp: /Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
src/rubiks/solvers/demos/solver_demo_cpp: src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/testing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable solver_demo_cpp"
	cd /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/solver_demo_cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/build: src/rubiks/solvers/demos/solver_demo_cpp
.PHONY : src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/build

src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/clean:
	cd /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos && $(CMAKE_COMMAND) -P CMakeFiles/solver_demo_cpp.dir/cmake_clean.cmake
.PHONY : src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/clean

src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/depend:
	cd /Users/oliver/ClionProjects/testing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/testing /Users/oliver/ClionProjects/testing/src/rubiks/solvers/demos /Users/oliver/ClionProjects/testing/build /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos /Users/oliver/ClionProjects/testing/build/src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/rubiks/solvers/demos/CMakeFiles/solver_demo_cpp.dir/depend

