#include "approximation.h"

UInt float_32_as_uint_32(const Float u)
{
    const union
    {
        Float f;
        UInt i;
    } fi = {u};

    return fi.i;
}

typedef const Float *restrict input;
typedef Float *restrict output;

#pragma omp declare simd
UInt get_table_index_from_float_format(const Float u)
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
UInt cap_index(const UInt b, const UInt cap)
{
    return b > cap ? cap : b;// Ensuring we don't overflow out of the table.
}
