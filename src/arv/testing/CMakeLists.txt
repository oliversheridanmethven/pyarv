## So we can find modules in the project root. (e.g. Criterion)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
# For C we quite like the Criterion framework.
find_package(Criterion REQUIRED)

add_library(testing INTERFACE)
target_include_directories(testing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(testing INTERFACE criterion)

add_subdirectory(tests)
