# Python bindings

We have setup this project to wrap some of the various C
libraries with Python bindings. To install these, run
```bash
cd build
pip3 install ..
```

This will call `scikit-build`, which in turn will invoke 
CMake, and build the whole project. After this, you can then 
also run the usual 
```bash
make
ctest
```