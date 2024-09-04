#ifndef TESTING_EXAMPLES_BINDINGS_H
#define TESTING_EXAMPLES_BINDINGS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject *multiply_into(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

#endif//TESTING_EXAMPLES_BINDINGS_H
