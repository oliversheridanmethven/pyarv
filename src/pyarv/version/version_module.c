#include "bindings/wrappers.h"
#include "version_bindings.h"

static PyMethodDef version_methods[] = {
        {"repo_name", PyFunc(repo_name_), METH_VARARGS | METH_KEYWORDS,
         "The repository's name."},
        {"repo_version", PyFunc(repo_version_), METH_VARARGS | METH_KEYWORDS,
         "The repository's version."},
        {"repo_author", PyFunc(repo_author_), METH_VARARGS | METH_KEYWORDS,
         "The repository's author."},
        {"repo_email", PyFunc(repo_email_), METH_VARARGS | METH_KEYWORDS,
         "The repository's email."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef version_module = {
        PyModuleDef_HEAD_INIT,
        "versions",
        "A simple module giving version information.",
        -1,
        version_methods};

PyMODINIT_FUNC
PyInit_version_bindings(void)
{
    PyObject *module = PyModule_Create(&version_module);
    if (!module)
    {
        fprintf(stderr, "Unable to create the version module.");
        return NULL;
    }

    return module;
}
