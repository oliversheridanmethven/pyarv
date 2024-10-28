#ifndef PYARV_CUBIC_H
#define PYARV_CUBIC_H

#include "approximation/approximation.h"

void cubic(const Float *restrict const input,
           Float *restrict const output,
           const size_t input_buffer_size);

#endif//PYARV_CUBIC_H
