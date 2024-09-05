#ifndef ARV_POLYNOMIAL_H
#define ARV_POLYNOMIAL_H

#include <stddef.h>

typedef unsigned int UInt;// We assume IEEE754
typedef float Float;      // We assume IEEE754

// For IEEE754
#define N_MANTISSA_32 23
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)

UInt float_32_as_uint_32(const Float u);
UInt get_table_index_from_float_format(const Float u);
UInt cap_index(const UInt b, const UInt cap);

void linear(const Float *restrict const input,
            Float *restrict const output,
            const size_t input_buffer_size);

void cubic(const Float *restrict const input,
           Float *restrict const output,
           const size_t input_buffer_size);

#endif//ARV_POLYNOMIAL_H
