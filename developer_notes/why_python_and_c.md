# Why Python and C?

This project is a Python wrapper of C implementations. 

## Why Python?

It's the original author's favourite language, so it seemed
a reasonable choice (as good as any other).

## Why C?

The original ACM TOMS paper provides mathematical recipes
for producing approximate random numbers, and the
speed improvements derive from exploiting SIMD hardware.
To go down to such a low-level brings us to the realms of
the usual compiled HPC languages such as 
C, C++, Fortran, Rust, etc. The original author didn't know 
Fortran or Rust very well, so that left us with C or C++. 

Choosing between C and C++ was a hard decision, and either 
would be a good platform to implement the underlying routines. 
So why C and not C++:

- The approximation uses some type punning (between `int` and `float`), 
and this is 
easily done in C using unions (and is defined behaviour), 
whereas in C++ type punning is much trickier to do correctly. 
- C has the keyword `restrict` which does not exist in C++ without 
compiler extensions (which we would prefer to avoid for portability),
and this is something we take advantage of for optimisation.
- C is the _lingua franca_ for HPC and portability, e.g. NumPy has a C API. 
- The original author slightly prefers C.

### Why not C++, Fortran, Rust, Julia, MATLAB, JavaScript, etc. 

Creating interfaces for the various
languages should be easy enough, and if the package proves
popular we welcome contributions to help us wide the scope.

For the higher level languages we hope to wrap our C code
similar to how we have for Python. For the lower level languages, either
wrappers, re-implement in the preferred language, or use the 
langauge feature, such as e.g. C++'s `extern "C"`. 