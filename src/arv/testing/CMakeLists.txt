## So we can find modules in the project root. (e.g. Criterion)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
# For C we quite like the Criterion framework.
find_package(Criterion REQUIRED)
include_directories(SYSTEM ${CRITERION_INCLUDE_DIRS})

add_library(testing INTERFACE)
target_include_directories(testing INTERFACE ${CRITERION_INCLUDE_DIRS})
target_include_directories(testing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(testing INTERFACE ${CRITERION_LIBRARIES})

add_subdirectory(tests)
