#define PY_SSIZE_T_CLEAN

#include "pyarv/bindings/examples/numpy_examples_bindings.h"
#include "arv/examples/numpy_examples.h"
#include <Python.h>
#include <iso646.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Python bindings */

PyObject *multiply_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return NULL;
}

PyObject *multiply_into_(PyObject *self, PyObject *args)
{
    double factor;
    size_t n;

    printf("Entering the function.");

    PyArrayObject *input_array;
    PyArrayObject *output_array;
    PyArg_ParseTuple(args, "O", &input_array);
    if (PyErr_Occurred())
    {
        PyErr_SetString(PyExc_RuntimeError, "An error seems to be have occurred.");
        return NULL;
    }
    printf("Completed the argument parsing.\n");
    //    Py_RETURN_NONE;
    if (not input_array)
    {
        PyErr_SetString(PyExc_TypeError, "The input array seems to be null.");
        return NULL;
    }
    printf("Checked for null arrays.\n");
    //    Py_RETURN_NONE;
    //    if (not PyArray_Check(input_array) or PyArray_TYPE(output_array) != NPY_DOUBLE)
    //    {
    //        PyErr_SetString(PyExc_TypeError, "The input array was not a numpy array of doubles.");
    //        return NULL;
    //    }
    //    printf("Checked first arrays.\n");
    //
    //    if (not PyArray_Check(output_array) or PyArray_TYPE(output_array) != NPY_DOUBLE)
    //    {
    //        PyErr_SetString(PyExc_TypeError, "The output array was not a numpy array of doubles.");
    //        return NULL;
    //    }
    printf("Checked arrays.\n");
    double *input = PyArray_DATA(input_array);
    int64_t input_size = PyArray_SIZE(input_array);
    Py_RETURN_NONE;
    double *output = nullptr;
    int64_t output_size = PyArray_SIZE(input_array);
    input = output = nullptr;
    if (input_size != output_size)
    {
        fprintf(stderr, "The input size (%lld) does not match the output size (%lld).", input_size, output_size);
        return NULL;
    }
    printf("Have input arrays of length: %lld", input_size);
    n = 0;
    factor = 2.0;
    printf("About to multiply.\n");
    Py_RETURN_NONE;

    multiply(input, output, n, factor);

    Py_RETURN_NONE;
}
