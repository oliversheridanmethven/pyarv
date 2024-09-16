add_library(gaussian linear.c cubic.c)
target_link_libraries(gaussian approximation)
target_include_directories(gaussian INTERFACE ..)