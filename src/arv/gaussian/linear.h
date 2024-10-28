#ifndef PYARV_LINEAR_H
#define PYARV_LINEAR_H

#include "approximation/approximation.h"

void linear(const Float *restrict const input,
            Float *restrict const output,
            const size_t input_buffer_size);

#endif//PYARV_LINEAR_H
