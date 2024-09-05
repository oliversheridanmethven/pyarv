#include "bindings/examples/simple_examples_bindings.h"
#include "bindings/wrappers.h"

static PyMethodDef examples_methods[] = {
        {"hello_world", PyFunc(hello_world_), METH_VARARGS | METH_KEYWORDS,
         "Says hello world."},
        {"foo", PyFunc(foo_), METH_VARARGS | METH_KEYWORDS,
         "Prints an arg and kwarg argument."},
        {"fatal_failure", PyFunc(fatal_failure_), METH_VARARGS | METH_KEYWORDS,
         "Fails and calls exit()."},
        {"non_fatal_failure", PyFunc(non_fatal_failure_), METH_VARARGS | METH_KEYWORDS,
         "Fails in a way Python can catch."},
        {"set_at_exit", PyFunc(set_at_exit_), METH_VARARGS | METH_KEYWORDS,
         "Sets functionality to execute on exiting."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef examples_module = {
        PyModuleDef_HEAD_INIT,
        "simple_examples",
        "A simple module giving examples to demonstrate C bindings.",
        -1,
        examples_methods};

PyMODINIT_FUNC
PyInit_simple_examples_bindings(void)
{
    PyObject *module = PyModule_Create(&examples_module);
    if (!module)
    {
        fprintf(stderr, "Unable to create the simple examples module.");
        return nullptr;
    }

    return module;
}
