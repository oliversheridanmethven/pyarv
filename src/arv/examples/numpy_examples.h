#ifndef TESTING_EXAMPLES_H
#define TESTING_EXAMPLES_H
#include "error_codes/error_codes.h"
#include <stddef.h>

void multiply(double *restrict const input,
              double *restrict const output,
              const size_t n,
              const double factor);

#endif//TESTING_EXAMPLES_H
