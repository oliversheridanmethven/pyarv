#include "numpy_examples_bindings.h"
// clang-format off
#define PY_ARRAY_UNIQUE_SYMBOL ARV_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// clang-format on

#include "bindings/wrappers.h"

PyMethodDef gaussian_methods[] = {
        {"multiply_into", PyFunc(multiply_into), METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL},
};

struct PyModuleDef gaussian_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_doc = "Examples of numpy bindings.",
        .m_name = "numpy_examples",
        .m_size = -1,
        .m_methods = gaussian_methods,
};

PyObject *
PyInit_numpy_examples_bindings(void)
{
    import_array();
    PyObject *module = PyModule_Create(&gaussian_module);
    if (
            !module || PyModule_AddStringConstant(module, "__version__", Py_STRINGIFY(NPB_VERSION)))
    {
        Py_XDECREF(module);
        return NULL;
    }
    return module;
}
