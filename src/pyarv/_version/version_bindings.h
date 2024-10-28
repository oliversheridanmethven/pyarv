#ifndef TESTING_VERSION_BINDINGS_H
#define TESTING_VERSION_BINDINGS_H

#include <Python.h>

PyObject *repo_name_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *repo_version_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *repo_author_(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *repo_email_(PyObject *self, PyObject *args, PyObject *kwargs);

#endif //TESTING_VERSION_BINDINGS_H
