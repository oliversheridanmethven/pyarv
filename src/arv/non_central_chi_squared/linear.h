#ifndef PYARV_LINEAR_H
#define PYARV_LINEAR_H

#include "approximation/approximation.h"

void linear(const Float *restrict const input,
            Float *restrict const output,
            const size_t input_buffer_size,
            const Float *restrict const non_centrality,
            const Float *restrict const degrees_of_freedom);

#endif//PYARV_LINEAR_H
