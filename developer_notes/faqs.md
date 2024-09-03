# Frequently asked questions

# I can't build with GCC on Mac OSX

On newer versions of Mac the latest versions of Clang use the
`libc++` implementation of the C++ standard library, whereas
GCC uses `libstdc++`, which does not have as much newer C++11
feature support. There is an ABI incompatability between the two, 
so on Mac something compiled with Clang can't be linked with something 
compiled with GCC, because they use two differing implementations 
of the C++ standard library. (On Linux both compilers use `libc++` so this
isn't as much of an issue).

Consequently, if on your Mac you install some standard libraries, such as 
e.g. Boost, which might contain non-header-only (e.g. program options), 
then on Mac then brew will likely default to Clang. This can start to lead
down a rabit hole of trying to compile everything with differing compilers
and standard library versions. 

Currently, the authors, who develop this on a Mac, haven't found a nice
solution for this. Consequently, on Mac, we recommend using the Clang 
compiler suite as it has better native support. 