cmake_minimum_required(VERSION 3.25)  # Use the latest and greatest.

# Declaring the project and its languages also sets the,
# compilers, so we set our paths to the compilers first.
# cf: https://stackoverflow.com/a/29904501/5134817

#find_program(CMAKE_C_COMPILER NAMES $ENV{CC} clang-19 clang PATHS ENV PATH NO_DEFAULT_PATH)
#find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc-14 PATHS ENV PATH NO_DEFAULT_PATH)

project(pyarv C)

# We want to use the latest standards.
# Only some of the latest and greatest compilers offer the appropriate level of support.
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)


set(CC ${CMAKE_C_COMPILER})
message(STATUS "We think we are using the C compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

set(CMAKE_GENERATOR "Unix Makefiles")

set(ERROR_FLAGS "-Wall -Wextra -Werror -Wattributes -Wunused-result -Wpedantic -pedantic")
if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Setting some extra Clang specific flags.")
    set(ERROR_FLAGS "${ERROR_FLAGS} -Wreserved-identifier")
elseif (CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Setting some extra AppleClang specific flags.")
    set(ERROR_FLAGS "${ERROR_FLAGS} -Wreserved-identifier")
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Setting some extra GNU specific flags.")
    set(ERROR_FLAGS "${ERROR_FLAGS}")
endif()

set(IGNORE_ERROR_FLAGS "-Wno-int-conversion -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-incompatible-pointer-types")
set(IGNORE_ERROR_FLAGS_C ${IGNORE_ERROR_FLAGS})
# Not all flags exist for both languages, so we possibly filter some out or add some in.
message(DEBUG "The CXX error flags before filtering: ${IGNORE_ERROR_FLAGS_CXX}")
set(C_ONLY_FLAGS "-Wno-int-conversion;-Wno-incompatible-pointer-types")

set(ERROR_FLAGS_C "${ERROR_FLAGS} ${IGNORE_ERROR_FLAGS_C}")
message(DEBUG "The C compiler flags are: ${CMAKE_C_FLAGS}")

message(STATUS "Compiling in mode: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Adding optimisation flags.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Adding debug flags.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
endif()

if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Setting some extra Clang specific flags.")
    set(OPENMP_FLAGS "-fopenmp")
elseif (CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Setting some extra AppleClang specific flags.")
    set(OPENMP_FLAGS "-Xclang -fopenmp")
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Setting some extra GNU specific flags.")
    set(OPENMP_FLAGS "-fopenmp")
endif()

set(CMAKE_C_FLAGS "-fPIC ${OPENMP_FLAGS} ${CMAKE_C_FLAGS} ${ERROR_FLAGS_C}")

cmake_host_system_information(RESULT N_CORES QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "We think we have ${N_CORES} cores on this machine.")
if(NOT N_CORES EQUAL 0)
    message(STATUS "Trying to set the build and tests to use parallel cores.")
    message(STATUS "Our default make tool is: ${CMAKE_MAKE_PROGRAM}")
endif()

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    set(INSOURCE_BUILD TRUE)
    message(STATUS "Producing an in-source build. (This is only recommended for developers!).")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif()

if(INSOURCE_BUILD)
    message(STATUS "Using an in source python path.")
    set(IN_SOURCE_PYTHON_PATH "${PROJECT_SOURCE_DIR}/src")
else()
    message(STATUS "Not using an in source python path.")
endif()

include(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG("-std=c99" COMPILER_SUPPORTS_C99)
if(COMPILER_SUPPORTS_C99)
    message(STATUS "Compiler ${CMAKE_C_COMPILER} appears to have C99 support.")
    foreach(i ${CMAKE_C_COMPILE_FEATURES})
    message(DEBUG "Compiler ${CMAKE_C_COMPILER} uses feature: ${i}")
    endforeach()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_C_COMPILER} has no C99 support.")
endif()

# To enable testing and use of ctest.
enable_testing()
# ^ This must be before any calls to add_subdirectory to ensure it is enabled recursively.
# cf. https://stackoverflow.com/questions/54550939/ctest-add-tests-in-subdirectories#comment109191766_55835742

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

# We use scikit-build to build our CMake project and setup our python wrappers.
if(SKBUILD)
    MESSAGE(STATUS "Trying to build using scikit-build")
    EXECUTE_PROCESS(COMMAND ${Python3_EXECUTABLE} -c "import importlib.resources; print(importlib.resources.files('skbuild').joinpath('resources/cmake'), end='')" OUTPUT_VARIABLE PYTHON_SKLIB_CMAKE_DIR)
    MESSAGE(STATUS "The Python scikit-build CMake directory is: ${PYTHON_SKLIB_CMAKE_DIR}")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PYTHON_SKLIB_CMAKE_DIR}/")
else()
    MESSAGE(STATUS "Could not find scikit-build for building.")
endif()

# We get the python libraries for our wrappers.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
MESSAGE(STATUS "Looking for the python libraries.")
# We will include and link to all the targets henceforth.
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

function(add_all_python_tests)
    file(GLOB_RECURSE python_files *.py)
    list(REMOVE_ITEM python_files ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)
    list(TRANSFORM python_files REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" OUTPUT_VARIABLE python_files_relative)
    message(DEBUG "The python files relative to the current cmake execution is: ${python_files_relative}")
    message(DEBUG "Possibly ignoring the file: ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")
    add_python_tests(${python_files_relative})
endfunction()

function(add_python_tests)
    foreach(python_file IN LISTS ARGN)
        add_python_test(${python_file})
    endforeach()
endfunction()

function(add_python_test python_file)
    STRING(REGEX REPLACE "\.py$" "" test_name ${python_file})
    message(DEBUG "Trying to add the python tests from: ${python_file} (${CMAKE_CURRENT_SOURCE_DIR}/${python_file})")
    add_test(NAME ${test_name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${python_file})
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PYTHONPATH=${IN_SOURCE_PYTHON_PATH}:${PYTHONPATH}")
endfunction()

function(install_python_target target destination)
    if(SKBUILD)
        MESSAGE(STATUS "Installing the Python target ${target} at destination: ${destination}.")
        target_link_libraries(${target} ${Python3_LIBRARIES})
        find_package(PythonExtensions REQUIRED)
        python_extension_module(${target})
        install(TARGETS ${target} LIBRARY DESTINATION ${destination})
    elseif (INSOURCE_BUILD)
        set(destination "${CMAKE_CURRENT_SOURCE_DIR}")
        MESSAGE(STATUS "Installing the Python target ${target} in source at destination: ${destination}.")
        target_link_libraries(${target} ${Python3_LIBRARIES})
        set_target_properties(${target} PROPERTIES PREFIX "") # Avoid the lib prefix which is required for python finding the module.
        install(TARGETS ${target} LIBRARY DESTINATION ${destination})
    endif()
endfunction()

# Where we find the source files for our builds.
add_subdirectory(src)

# Where we find the scratch files for our proto-typing things.
#add_subdirectory(misc)
