#ifndef TESTING_WRAPPERS_H
#define TESTING_WRAPPERS_H

#define PyFunc(function)                                                                                 \
    /* An awkward cast necessary for functions of the form def foo(*args, **kwargs) */                   \
    /* https://docs.python.org/3/extending/extending.html#keyword-parameters-for-extension-functions/ */ \
    ((PyCFunction) (void (*)(void)) function)

#endif //TESTING_WRAPPERS_H
