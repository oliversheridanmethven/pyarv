#include "numpy_examples.h"
#include <stdio.h>

void multiply(double *restrict const input,
              double *restrict const output,
              const size_t n,
              const double factor)
{
    for (size_t i = 0; i < n; i++)
    {
        output[i] = input[i] * factor;
    }
}
