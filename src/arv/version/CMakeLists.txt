# So we can add versioning to our repo
find_package(Git)
set(generated_version_source "${CMAKE_CURRENT_SOURCE_DIR}/version.c")
add_custom_command(
        OUTPUT ${generated_version_source}
        COMMAND ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/version.c.in
        -D DST=${generated_version_source}
        -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -D CMAKE_PROJECT_NAME=${CMAKE_PROJECT_NAME}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/GenerateVersionHeader.cmake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


add_library(version ${generated_version_source})
target_include_directories(version INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
