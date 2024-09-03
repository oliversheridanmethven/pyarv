#define PY_SSIZE_T_CLEAN

#include "../../arv/examples.h"
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

/* Python bindings */

PyObject *hello_world_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (hello_world())
    {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject *foo_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    /* This is the python equivalent of: def foo(a, b="default", **kwargs): ...  */
    static char *keywords[] = {"a", "b", NULL};
    int a;
    char *b = "default";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|s:foo_", keywords, &a, &b))
    {
        /* The arguments passed don't correspond to the signature described. */
        return NULL;
    }

    if (foo(a, b))
    {
        return nullptr;
    }
    return PyLong_FromLong(a);
}

static void exit_handler(void)
{
    fprintf(stderr, "Calling the exit handler from C.\n");
}

static void set_at_exit(void)
{
    atexit(&exit_handler);
}

PyObject *set_at_exit_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    set_at_exit();
    Py_RETURN_NONE;
}

PyObject *fatal_failure_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    fatal_failure();
    Py_RETURN_NONE;
}

PyObject *non_fatal_failure_(PyObject *self, PyObject *args, PyObject *kwargs)
{
    fprintf(stderr, "A non-fatal error from C in our Python binding.\n");
    PyErr_Format(PyExc_Exception, "Raising a non-fatal error from C in our Python binding up to Python.\n");
    return nullptr;
}
