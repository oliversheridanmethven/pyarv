#ifndef TESTING_EXAMPLES_BINDINGS_H
#define TESTING_EXAMPLES_BINDINGS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject *polynomial_(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs);

#endif//TESTING_EXAMPLES_BINDINGS_H
