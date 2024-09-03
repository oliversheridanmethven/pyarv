#ifndef TESTING_EXAMPLES_H
#define TESTING_EXAMPLES_H
#include "../../../../testing/src/error_codes/error_codes.h"

[[nodiscard]]
error_code hello_world(void);

[[nodiscard]]
error_code foo(int a, char *b);

void fatal_failure(void);

#endif//TESTING_EXAMPLES_H
