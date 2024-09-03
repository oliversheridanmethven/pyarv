#ifndef TESTING_EXAMPLES_BINDINGS_H
#define TESTING_EXAMPLES_BINDINGS_H

#include <Python.h>

PyObject *hello_world_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *foo_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *fatal_failure_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *non_fatal_failure_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *set_at_exit_(PyObject *self, PyObject *args, PyObject *kwargs);

#endif//TESTING_EXAMPLES_BINDINGS_H
