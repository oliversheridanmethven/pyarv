#ifndef PYARV_APPROXIMATION_H
#define PYARV_APPROXIMATION_H

#include <iso646.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/* It is common for implementations to use IEEE-754, and be compliant enough for our purposes,
 * but not be fully compliant (or to be fully compliant and not declare itself so). Consequently
 * for the following headers, while we could be really strict, we do a best effort to check
 * that the compiler seems to be IEEE-754 compliant. 
 */

static_assert(__STDC_VERSION__ >= 202311L, "C23 or newer is required.");

#if defined(__STDC_IEC_559__)
    static_assert(__STDC_IEC_559__ == 1, "Floats must conform to IEEE-754.")
#endif
#if defined(__STDC_IEC_60559_BFP__)
    static_assert(__STDC_IEC_60559_BFP__ >= 202311L, "Floats must conform to IEEE-754."); // cf. https://stackoverflow.com/a/31967139/5134817 
#endif

typedef unsigned int UInt;// We assume IEEE754
typedef float Float;      // We assume IEEE754

// For IEEE754
#define N_MANTISSA_32 23
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)

static_assert(sizeof(char) == 1, "A char should be 1 byte"); // A sanity check but this is specified by the language.
static_assert(CHAR_BIT == 8, "A byte should be 8 bits");
static_assert(sizeof(float) == 4, "32-bit (4 byte) floats are requried.");
static_assert(FLT_RADIX == 2, "Floats must use base 2.");
static_assert(FLT_MANT_DIG == N_MANTISSA_32 + 1, "Floats must have the expected number of mantissa digits."); // cf. https://en.cppreference.com/w/cpp/types/climits (+1 for the implicit digit)
static_assert(FLT_MAX_EXP == FLOAT32_EXPONENT_BIAS + 1, "Floats must have the expected number of mantissa digits."); // cf. https://en.cppreference.com/w/cpp/types/climits 
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
