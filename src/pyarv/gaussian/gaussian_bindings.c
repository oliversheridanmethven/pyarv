// clang-format off
#include "gaussian_bindings.h"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PYARV_GAUSSIAN_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// clang-format on

#include "gaussian/cubic.h"
#include "gaussian/linear.h"

//TODO: Remove code duplication, perhaps by a macro?
PyObject *
linear_(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *input_array;
    PyArrayObject *output_array;
#define N_ARRAYS 2
    PyArrayObject **arrays[N_ARRAYS] = {&input_array, &output_array};
    char *arg_names[] = {
            "inputs",
            "outputs",
            NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "$O!O!:linear",
                                     arg_names,
                                     &PyArray_Type,
                                     &input_array,
                                     &PyArray_Type,
                                     &output_array))
    {
        return NULL;
    }
    for (int i = 0; i < N_ARRAYS; i++)
    {
        PyObject *array = *arrays[i];
        if (PyArray_NDIM(array) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional.");
            return NULL;
        }
        if (PyArray_TYPE(array) != NPY_FLOAT32)
        {
            PyErr_SetString(PyExc_ValueError, "Array must be of type float32.");
            return NULL;
        }

        if (!PyArray_IS_C_CONTIGUOUS(array))
        {
            PyErr_SetString(PyExc_ValueError, "Array must be C contiguous.");
            return NULL;
        }
    }

    npy_float32 *input_buffer = (npy_float32 *) PyArray_DATA(input_array);
    npy_float32 *output_buffer = (npy_float32 *) PyArray_DATA(output_array);
    size_t input_buffer_size = PyArray_SIZE(input_array);
    size_t output_buffer_size = PyArray_SIZE(output_array);

    if (input_buffer_size != output_buffer_size)
    {
        PyErr_SetString(PyExc_ValueError, "The input and output arrays are of differing lengths.");
        return NULL;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS; /* No longer need the Python GIL */

    linear(input_buffer, output_buffer, input_buffer_size);

    NPY_END_THREADS; /* We return the Python GIL. */

    Py_RETURN_NONE;
}

PyObject *
cubic_(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *input_array;
    PyArrayObject *output_array;
#define N_ARRAYS 2
    PyArrayObject **arrays[N_ARRAYS] = {&input_array, &output_array};
    char *arg_names[] = {
            "inputs",
            "outputs",
            NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "$O!O!:linear",
                                     arg_names,
                                     &PyArray_Type,
                                     &input_array,
                                     &PyArray_Type,
                                     &output_array))
    {
        return NULL;
    }
    for (int i = 0; i < N_ARRAYS; i++)
    {
        PyObject *array = *arrays[i];
        if (PyArray_NDIM(array) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional.");
            return NULL;
        }
        if (PyArray_TYPE(array) != NPY_FLOAT32)
        {
            PyErr_SetString(PyExc_ValueError, "Array must be of type float32.");
            return NULL;
        }

        if (!PyArray_IS_C_CONTIGUOUS(array))
        {
            PyErr_SetString(PyExc_ValueError, "Array must be C contiguous.");
            return NULL;
        }
    }

    npy_float32 *input_buffer = (npy_float32 *) PyArray_DATA(input_array);
    npy_float32 *output_buffer = (npy_float32 *) PyArray_DATA(output_array);
    size_t input_buffer_size = PyArray_SIZE(input_array);
    size_t output_buffer_size = PyArray_SIZE(output_array);

    if (input_buffer_size != output_buffer_size)
    {
        PyErr_SetString(PyExc_ValueError, "The input and output arrays are of differing lengths.");
        return NULL;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS; /* No longer need the Python GIL */

    cubic(input_buffer, output_buffer, input_buffer_size);

    NPY_END_THREADS; /* We return the Python GIL. */

    Py_RETURN_NONE;
}
