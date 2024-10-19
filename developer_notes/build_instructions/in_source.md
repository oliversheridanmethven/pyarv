# In source builds


## Why allow in source builds?

In source builds are largely discouraged, and are only 
recommended for use by developers. The reason we support this 
for developers is to populate the source directory with various 
generated files, libraries, etc. The use of this is for example:
placing python extension libraries in the source directory. 
This allowes for them to be picked up by an interpreter
which can point to the in source code, rather than what is 
produced by the scikit build procedure. This means code
which is under development can be more easily accessed by
an IDE or interpreter. 

## Making in source builds

From the projects root directory run 
```bash
cmake .
make 
make install
make test 
```

!!! note 
    This is run from the project's root directory, not from a 
    separate dedicated build directory.

!!! note "IDES"
    Adding `-DINSOURCE_BUILD=True` to the `cmake .` can allow the 
    IDE to propagate the appropriate source path, which is useful for 
    e.g. running the tests. 


# Cleaning everything up

To clean everything up, run 
```bash
make clean
./cmake_uninstall.sh
```

### Adjusting your `PYTHONPATH`

If you plan on testing out the code using the source code in 
the repository, such as for use in a terminal setting or an 
IDE, then it might be useful to add the contents of `src/` to
your `PYTHONPATH` environment variable. To do just this we
have the script `add_src_to_python_path.sh` which can be run 
from the project's root directory by calling:
```bash
source add_src_to_python_path.sh
```