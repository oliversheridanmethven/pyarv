#ifndef PYARV_APPROXIMATION_H
#define PYARV_APPROXIMATION_H

#include <stddef.h>
#include <stdlib.h>
#if __STDC_VERSION__ < 202311L
#include <stdbool.h>
#endif
#include <iso646.h>

typedef unsigned int UInt;// We assume IEEE754
typedef float Float;      // We assume IEEE754

// For IEEE754
#define N_MANTISSA_32 23
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)

/*
 * NB - The static in the function signatures below seems required
 * in my newest CMake setup (very fragile and hacky, I know...),
 * else on my Mac I get complaints about the symbols not existing
 * the flat name space. Seems to have happened after some compiler
 * and OS upgrades. 
 */

union FloatAndInt
{
    Float f;
    UInt i;
};

#pragma omp declare simd
static inline UInt float_32_as_uint_32(const Float u)
{
    const union FloatAndInt fi = {u};
    return fi.i;
}

#pragma omp declare simd
static inline UInt get_table_index_from_float_format(const Float u)
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
static inline UInt cap_index(const UInt b, const UInt cap)
{
    return b > cap ? cap : b;// Ensuring we don't overflow out of the table.
}

#pragma omp declare simd
static inline Float ternary(bool p, const Float t, const Float f) {
    /*
     * Branches (and hence ternary expressions) can trip up some vectorisation routines.
     * This is a version which selects from both branches results as appropriate
     * which is doesn't trip up most vectorising compilers. 
     */
    return p * t + (not p) * f;
}

#endif//PYARV_APPROXIMATION_H
