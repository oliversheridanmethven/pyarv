# Why Python and C?

This project is a Python wrapper of C implementations. 

## Python

It's the original author's favourite language, so it seemed
a reasonable choice (as good as any other).

## C

The original ACM TOMS paper provides mathematical recipes
for producing approximate random numbers, and the
speed improvements derive from exploiting SIMD hardware.
To go down to such a low-level brings us to the realms of
the usual compiled HPC languages such as 
C, C++, Fortran, Rust, etc. The original author doesn't know 
Fortran or Rust very well, so that leaves us with C or C++. 

Choosing between C and C++ was a hard decision, and either 
would be a good platform to implement the underlying routines. 
So why C and not C++:

- The approximations use some type punning (between `int` and `float`), 
and this is 
easily done in C using unions (and is defined behaviour in C, including C99 onwards, 
cf. https://stackoverflow.com/a/25664954/5134817 and https://stackoverflow.com/a/25672839/5134817), 
whereas in C++ type punning is much trickier to do correctly 
(without producing undefined behaviour). 
- C has the keyword `restrict` which does not exist in C++ without 
compiler extensions (which we would prefer to avoid for portability),
and this is something we take advantage of for optimisation.
- C is the _lingua franca_ for HPC and portability.
- Python and NumPy have C APIs, and these are the initial 
targets for our wrappers.

## Why not C++, Fortran, Rust, Julia, MATLAB, JavaScript, etc. 

Creating interfaces for the various
languages should be easy enough, and if the package proves
popular we welcome contributions to help us widen the scope.

For the higher level languages we hope to wrap our C code
similar to how we have for Python. For the lower level languages, 
either use
wrappers, re-implement in the preferred language, or use 
langauge features, such as e.g. C++'s `extern "C"`. 