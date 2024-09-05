add_library(gaussian polynomial.c linear.c cubic.c)
target_include_directories(gaussian INTERFACE ..)