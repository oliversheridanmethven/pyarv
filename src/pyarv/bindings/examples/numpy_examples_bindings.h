#ifndef TESTING_EXAMPLES_BINDINGS_H
#define TESTING_EXAMPLES_BINDINGS_H

#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *multiply_(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *multiply_into_(PyObject *self, PyObject *args);

#endif//TESTING_EXAMPLES_BINDINGS_H
