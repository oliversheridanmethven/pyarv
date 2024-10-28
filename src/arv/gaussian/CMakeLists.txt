add_library(gaussian linear.c cubic.c)
target_link_libraries(gaussian approximation Python3::NumPy)
target_include_directories(gaussian INTERFACE ..)