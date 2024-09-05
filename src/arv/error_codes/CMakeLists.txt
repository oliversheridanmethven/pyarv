add_subdirectory(tests)

add_library(error_codes error_codes.c)
target_include_directories(error_codes INTERFACE ..)
