# Building from source

## Out-of-source builds 

We follow the CMake convention by only encouraging "out-of-source"
builds, hence the reason for this build directory existing.
This directory exists only for manual building and testing with `cmake` et al.

To build the project:
```bash
cd build
cmake .. 
make 
pip3 install .. 
ctest 
```
!!! note
    `make test` and `ctest` are synonymous.

!!! note "Running things in parallel"
    `cmake`, `ctest`, and `make` can all use multiple cores, 
    to speed things up, typically by adding a `-j <N>` or 
    `--parallel <N>` flag. 

The reason we install the Python package is because many of our
tests are Python based as well, including several 
C extensions and various Python modules we have created. 
Hence, without this step, the various Python imports won't work
correctly and many of the tests can be expected to fail. 

### Modern C23 compilers

We are trying to use a very modern C standard
(C23 is quite new at the time of writing), and compiler
support for this is limited. To ensure `cmake` can find
a sufficiently new compiler version, it may be necessary to
hardwire paths to these in your invocation of `cmake`, e.g.:

```bash
cmake -D CMAKE_C_COMPILER=/usr/local/Cellar/gcc/13.1.0/bin/gcc-13 -D CMAKE_CXX_COMPILER=/usr/local/Cellar/gcc/13.1.0/bin/g++-13 ..
```

## Debugging

If any tests are failing, then these
can be debugged further by running
```bash
ctest --rerun-failed --output-on-failure
```
