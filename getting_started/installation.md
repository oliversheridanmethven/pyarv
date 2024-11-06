# Installation

To install the most recent release run:
```bash
pip install pyarv
```
or to install the latest version run:
```bash
pip install git+https://github.com/oliversheridanmethven/pyarv.git
```

To test the installation is working fine run:
```bash
pytest --pyargs pyarv
```

## Common installation issues

### Apple Clang does not support `-fopenmp`

Apple Clang does not support `-fopenmp` out of the box,
but a simple `brew install [gcc|llvm]` resolves this issue by installing
a nice new compiler which does support the flag. 
If you are required to use the
native Apple Clang compiler, then doing `brew reinstall libomp`
should circumvent the issue.

### GCC on Mac complains about missing definitions and unknown types from `stdio.h`

Trying to use modern versions of GCC (e.g. 14.2) 
on Mac complains about missing standard
library functions, types, and definitions from `stdio.h`,
such as:  
`error: unknown type name 'FILE'`  

There is an incompatability between CMake, Mac, XCode, and
GCC, which is [documented here with a MWE](https://discourse.cmake.org/t/issue-between-cmake-mpi-macos-xcode-16-and-gcc/11711/9).
A solution to this is to
```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk/
```
before calling CMake. Note that newer versions of the SDK, such as 15,
still exhibit this issue.
(Perhaps put this in your `~/.bashrc`, `~/.zshrc`, or similar.)