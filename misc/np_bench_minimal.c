# define PY_SSIZE_T_CLEAN
# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL NPB_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"

static struct PyModuleDef npb_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "np_bench",
    .m_size = -1,
};

PyObject *
PyInit_np_bench(void)
{
    import_array();
    PyObject *m = PyModule_Create(&npb_module);
    if (!m || PyModule_AddStringConstant(m, "__version__", "0.1.0")
    ) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

