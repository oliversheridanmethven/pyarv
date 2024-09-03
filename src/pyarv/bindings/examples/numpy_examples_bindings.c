#define PY_SSIZE_T_CLEAN

#include "pyarv/bindings/examples/numpy_examples_bindings.h"
#include "arv/examples/numpy_examples.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

/* Python bindings */

PyObject *multiply_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    /* This is the python equivalent of: def foo(a, b="default", **kwargs): ...  */
    static char *keywords[] = {"input", "output", "factor", "n", NULL};
    double *input, *output;
    double factor;
    size_t n;

    // TODO: ...

    input = output = nullptr;
    n = 0;
    factor = 2.0;
    multiply(input, output, n, factor);

    Py_RETURN_NONE;
}
