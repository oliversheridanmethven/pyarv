#include "pyarv/bindings/examples/numpy_examples_bindings.h"
#include "pyarv/bindings/wrappers.h"

static PyMethodDef examples_methods[] = {
        {"multiply", PyFunc(multiply_), METH_VARARGS | METH_KEYWORDS,
         "Multiplies one vector by some factor."},
        {"multiply_into", multiply_into_, METH_VARARGS,
         "Multiplies one vector by some factor, storing the result into another."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef examples_module = {
        PyModuleDef_HEAD_INIT,
        "numpy_examples",
        "A simple module giving numpy examples to demonstrate C bindings.",
        -1,
        examples_methods};

PyMODINIT_FUNC
PyInit_numpy_examples_bindings(void)
{
    import_array();
    PyObject *module = PyModule_Create(&examples_module);
    if (!module)
    {
        fprintf(stderr, "Unable to create the numpy examples module.");
        return nullptr;
    }
    import_array();
    return module;
}
