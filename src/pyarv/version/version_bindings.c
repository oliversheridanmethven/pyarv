#include "version_bindings.h"
#include "version/version.h"

PyObject *repo_name_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return PyUnicode_FromString(repo_name());
}

PyObject *repo_version_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return PyUnicode_FromString(repo_version());
}

PyObject *repo_author_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return PyUnicode_FromString(repo_author());
}

PyObject *repo_email_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return PyUnicode_FromString(repo_email());
}
