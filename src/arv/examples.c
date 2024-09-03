#include "examples.h"
#include <stdio.h>
#include <stdlib.h>

error_code hello_world(void)
{
    int rc = puts("Hello world from within C.");
    return (rc == EOF || rc < 0) ? EC_FAILURE : EC_SUCCESS;
}

error_code foo(int a, char *b)
{
    int rc = printf("The input values are: a = %i and b = %s\n", a, b);
    return (rc >= 0) ? EC_SUCCESS : EC_FAILURE;
}


[[noreturn]] void fatal_failure(void)
{
    fprintf(stderr, "We are fatally failing now.\n");
    exit(1);
}
