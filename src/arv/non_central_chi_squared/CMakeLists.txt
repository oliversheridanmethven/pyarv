add_library(non_central_chi_squared polynomial.c linear.c cubic.c)
target_include_directories(non_central_chi_squared INTERFACE ..)