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
CMAKE_SOURCE_DIR = /Users/oliver/ClionProjects/pyarv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/oliver/ClionProjects/pyarv

# Include any dependencies generated for this target.
include src/arv/version/CMakeFiles/version.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/arv/version/CMakeFiles/version.dir/compiler_depend.make

# Include the progress variables for this target.
include src/arv/version/CMakeFiles/version.dir/progress.make

# Include the compile flags for this target's objects.
include src/arv/version/CMakeFiles/version.dir/flags.make

src/arv/version/version.c:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/oliver/ClionProjects/pyarv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating version.c"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && /usr/local/Cellar/cmake/3.29.2/bin/cmake -D SRC=/Users/oliver/ClionProjects/pyarv/src/arv/version/version.c.in -D DST=/Users/oliver/ClionProjects/pyarv/src/arv/version/version.c -D GIT_EXECUTABLE=/usr/local/bin/git -D CMAKE_PROJECT_NAME=pyarv -P /Users/oliver/ClionProjects/pyarv/src/arv/version/GenerateVersionHeader.cmake

src/arv/version/CMakeFiles/version.dir/version.c.o: src/arv/version/CMakeFiles/version.dir/flags.make
src/arv/version/CMakeFiles/version.dir/version.c.o: src/arv/version/version.c
src/arv/version/CMakeFiles/version.dir/version.c.o: src/arv/version/CMakeFiles/version.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/oliver/ClionProjects/pyarv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/arv/version/CMakeFiles/version.dir/version.c.o"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/arv/version/CMakeFiles/version.dir/version.c.o -MF CMakeFiles/version.dir/version.c.o.d -o CMakeFiles/version.dir/version.c.o -c /Users/oliver/ClionProjects/pyarv/src/arv/version/version.c

src/arv/version/CMakeFiles/version.dir/version.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/version.dir/version.c.i"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/oliver/ClionProjects/pyarv/src/arv/version/version.c > CMakeFiles/version.dir/version.c.i

src/arv/version/CMakeFiles/version.dir/version.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/version.dir/version.c.s"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && /opt/local/bin/clang-mp-16 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/oliver/ClionProjects/pyarv/src/arv/version/version.c -o CMakeFiles/version.dir/version.c.s

# Object files for target version
version_OBJECTS = \
"CMakeFiles/version.dir/version.c.o"

# External object files for target version
version_EXTERNAL_OBJECTS =

lib/libversion.a: src/arv/version/CMakeFiles/version.dir/version.c.o
lib/libversion.a: src/arv/version/CMakeFiles/version.dir/build.make
lib/libversion.a: src/arv/version/CMakeFiles/version.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/oliver/ClionProjects/pyarv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C static library ../../../lib/libversion.a"
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && $(CMAKE_COMMAND) -P CMakeFiles/version.dir/cmake_clean_target.cmake
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/version.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/arv/version/CMakeFiles/version.dir/build: lib/libversion.a
.PHONY : src/arv/version/CMakeFiles/version.dir/build

src/arv/version/CMakeFiles/version.dir/clean:
	cd /Users/oliver/ClionProjects/pyarv/src/arv/version && $(CMAKE_COMMAND) -P CMakeFiles/version.dir/cmake_clean.cmake
.PHONY : src/arv/version/CMakeFiles/version.dir/clean

src/arv/version/CMakeFiles/version.dir/depend: src/arv/version/version.c
	cd /Users/oliver/ClionProjects/pyarv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/arv/version /Users/oliver/ClionProjects/pyarv /Users/oliver/ClionProjects/pyarv/src/arv/version /Users/oliver/ClionProjects/pyarv/src/arv/version/CMakeFiles/version.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/arv/version/CMakeFiles/version.dir/depend

