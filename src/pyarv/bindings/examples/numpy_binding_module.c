#define PY_SSIZE_T_CLEAN
#include <Python.h>
// clang-format off
#define PY_ARRAY_UNIQUE_SYMBOL ARV_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// clang-format on
#include "pyarv/bindings/examples/numpy_examples_bindings.h"

PyObject *
multiply_into(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *input_array;
    PyArrayObject *output_array;
    double factor;
#define N_ARRAYS 2
    PyArrayObject **arrays[N_ARRAYS] = {&input_array, &output_array};
    char *arg_names[] = {
            "input",
            "output",
            "factor",
            NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "$O!O!d:multiply",
                                     arg_names,
                                     &PyArray_Type,
                                     &input_array,
                                     &PyArray_Type,
                                     &output_array,
                                     &factor))
    {
        return NULL;
    }
    for (int i = 0; i < N_ARRAYS; i++)
    {
        PyObject *array = *arrays[i];
        if (PyArray_NDIM(array) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
            return NULL;
        }
        if (PyArray_TYPE(array) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_ValueError, "Array must be of type double");
            return NULL;
        }

        if (!PyArray_IS_C_CONTIGUOUS(array))
        {
            PyErr_SetString(PyExc_ValueError, "Array must be C contiguous.");
            return NULL;
        }
    }

    npy_double *input_buffer = (npy_double *) PyArray_DATA(input_array);
    npy_double *output_buffer = (npy_double *) PyArray_DATA(output_array);
    size_t input_buffer_size = PyArray_SIZE(input_array);
    size_t output_buffer_size = PyArray_SIZE(output_array);

    if (input_buffer_size != output_buffer_size)
    {
        PyErr_SetString(PyExc_ValueError, "The input and output arrays are of differing lengths.");
        return NULL;
    }

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS; /* No longer need the Python GIL */

    for (size_t i = 0; i < input_buffer_size; i++)
    {
        output_buffer[i] = input_buffer[i] * factor;
    }

    NPY_END_THREADS; /* We return the Python GIL. */

    Py_RETURN_NONE;
}

PyMethodDef example_methods[] = {
        {"multiply_into", (PyCFunction) multiply_into, METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL},
};

struct PyModuleDef example_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_doc = "Examples of numpy bindings.",
        .m_name = "numpy_examples",
        .m_size = -1,
        .m_methods = example_methods,
};

PyObject *
PyInit_numpy_examples_bindings(void)
{
    import_array();
    PyObject *module = PyModule_Create(&example_module);
    if (
            !module || PyModule_AddStringConstant(module, "__version__", Py_STRINGIFY(NPB_VERSION)))
    {
        Py_XDECREF(module);
        return NULL;
    }
    return module;
}
