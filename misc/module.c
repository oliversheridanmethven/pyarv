#include "module_headers.h"

#define PY_ARRAY_UNIQUE_SYMBOL EXAMPLE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyMethodDef gaussian_methods[] = {
        {"foo", (PyCFunction) (void (*)(void)) foo, METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL},
};

static struct PyModuleDef gaussian_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_doc = "Something is going wrong here.",
        .m_name = "examples",
        .m_size = -1,
        .m_methods = gaussian_methods,
};

PyObject *
PyInit_module_examples(void)
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
