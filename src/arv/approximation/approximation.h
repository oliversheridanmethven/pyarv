#ifndef PYARV_APPROXIMATION_H
#define PYARV_APPROXIMATION_H

#include <stddef.h>

typedef unsigned int UInt;// We assume IEEE754
typedef float Float;      // We assume IEEE754

// For IEEE754
#define N_MANTISSA_32 23
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)

union FloatAndInt
{
    Float f;
    UInt i;
};

#pragma omp declare simd
inline UInt float_32_as_uint_32(const Float u)
{
    const union FloatAndInt fi = {u};
    return fi.i;
}

typedef const Float *restrict input;
typedef Float *restrict output;

#pragma omp declare simd
inline UInt get_table_index_from_float_format(const Float u)
{
    /*
     * Takes the approximate logarithm of a floating point number and maps this to
     * an array index, where we have the following mappings:
     *
     *      Input value/range           Output Index
     *      0.5                 ->          0
     *      [0.25,      0.5)    ->          1
     *      [0.125,     0.25)   ->          2
     *      ...                             ...
     *                                      14
     *                                      15
     *                                      16
     *                                      ...
     *
     * Assumes input has a zero in its sign bit.
     *
     */

    UInt b;
    b = float_32_as_uint_32(u) >> N_MANTISSA_32;// Keeping only the exponent, and removing the mantissa.
    b = FLOAT32_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    return b;
}

#pragma omp declare simd
inline UInt cap_index(const UInt b, const UInt cap)
{
    return b > cap ? cap : b;// Ensuring we don't overflow out of the table.
}


#endif//PYARV_APPROXIMATION_H
