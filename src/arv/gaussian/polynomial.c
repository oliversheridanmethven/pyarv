#include "polynomial.h"

#define TABLE_SIZE 16
#define TABLE_MAX_INDEX (TABLE_SIZE - 1)// Zero indexing...

//#define FLOAT32_AS_UINT32(x) (*((UInt *) &x))
static inline UInt FLOAT32_AS_UINT32(const Float u)
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

const Float poly_coef_0[TABLE_SIZE] = {0x0.0p+0, -0x1.ac912ee8e02c7p+0, -0x1.f75e814149e51p+0, -0x1.1e6180aef5ee5p+1, -0x1.3e6bdc930dbf5p+1, -0x1.5c4345a1c8502p+1, -0x1.783f61171b526p+1, -0x1.92a296c80960ep+1, -0x1.aba9aa34f57abp+1, -0x1.c38801fda2ad3p+1, -0x1.da5d1882c4109p+1, -0x1.f04612cea4e4ep+1, -0x1.02ad8bfca2f21p+2, -0x1.0941e2cc18ad4p+2, 0x0.0p+0, 0x0.0p+0};
const Float poly_coef_1[TABLE_SIZE] = {0x0.0p+0, 0x1.482332981b5c8p+2, 0x1.146c729d29385p+3, 0x1.e59802f5a5569p+3, 0x1.b45b568784d99p+4, 0x1.8eba299142809p+5, 0x1.70e2de5a2468dp+6, 0x1.5864ee7401797p+7, 0x1.43e98a100b1dep+8, 0x1.32ab14bed3b48p+9, 0x1.23d09d9151873p+10, 0x1.16d0052206dadp+11, 0x1.0b47c49826650p+12, 0x1.99e1f9ac0cdcfp+12, 0x0.0p+0, 0x0.0p+0};
const Float poly_coef_2[TABLE_SIZE] = {0x0.0p+0, -0x1.5d5cd981445c6p+2, -0x1.3d93ba35dd604p+4, -0x1.264a3456e23b5p+6, -0x1.10d123594e05bp+8, -0x1.fca7e635174aap+9, -0x1.dd16d660e56f5p+11, -0x1.c1a240ead4a0dp+13, -0x1.a9c2af86b2f0dp+15, -0x1.956196ef09060p+17, -0x1.838482eeaa98ap+19, -0x1.738fb92981eafp+21, -0x1.6525f6cf2cc81p+23, -0x1.b6488712540bcp+24, 0x0.0p+0, 0x0.0p+0};
const Float poly_coef_3[TABLE_SIZE] = {0x0.0p+0, 0x1.e6c176d279f78p+1, 0x1.7e068b63c7d33p+4, 0x1.5c2c985caca52p+7, 0x1.428ee90b0f3fcp+10, 0x1.2d57737bd1fa9p+13, 0x1.1b4f3b1574993p+16, 0x1.0b70efc6690e9p+19, 0x1.faff5cc94f54ap+21, 0x1.e358595675accp+24, 0x1.ce9884b53ebf5p+27, 0x1.bbe322ecbca45p+30, 0x1.aad9f957d0f37p+33, 0x1.a9124cabf7fdcp+35, 0x0.0p+0, 0x0.0p+0};

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static inline Float polynomial_approximation(Float u, UInt b)
{
/*
     * Polynomial approximation of a function.
     *
     * This assumes a very small polynomial and will exploit Horner's rule
     * and splits the polynomial into the even and odd terms. This could
     * (and should) be generalised to bigger polynomials, but more ideally
     * tailored to constants, piecewise linear, and quadratic. Beyond cubic
     * and this might be redundant and then a very generalised implementation
     * might be preferable...
     *
     * Input:
     *      u - Input position.
     *      b - Index of polynomial coefficient to use.
     *
     */

// This would ideally be implemented using vector registers and scatter/gather intrinsics.
#if (POLYNOMIAL_ORDER == 3)
    Float z, z_even, z_odd;
    //  return poly_coef_0[b] + u * (poly_coef_1[b] + u * (poly_coef_2[b] + u * poly_coef_3[b]));
    Float x = u * u;
    z_even = poly_coef_0[b] + poly_coef_2[b] * x;
    z_odd = poly_coef_1[b] + poly_coef_3[b] * x;
    z = z_even + z_odd * u;
    return z;
#endif
}

static inline UInt get_table_index_from_float_format(Float u)
{
    /*
     * Takes the approximate logarithm of a floating point number and maps this to
     * an array index, where we have the following mappings:
     *
     *      Input value/range           Output Index
     *      0.5                 ->          0
     *      [0.25,      0.5)    ->          1
     *      [0.125,     0.25)   ->          2
     *      ...
     *                                      14
     *                                      15
     *                                      15  <<  Table is capped at 16 entries
     *                                      15  <<
     *                                      15  <<
     *                                      ...
     *                                      15  <<
     *
     * Assumes input has a zero in its sign bit.
     *
     */

    UInt b;
    b = FLOAT32_AS_UINT32(u) >> N_MANTISSA_32;    // Keeping only the exponent, and removing the mantissa.
    b = FLOAT32_EXPONENT_BIAS_TABLE_OFFSET - b;   // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;// Ensuring we don't overflow out of the table.
    return b;
}

void polynomial(const Float *restrict const input,
                Float *restrict const output,
                const size_t input_buffer_size)
{
    for (size_t i = 0; i < input_buffer_size; i++)
    {
        Float x, z;
        x = input[i];
        bool predicate = x < 0.5f;
        x = predicate ? x : 1.0f - x;
        UInt b = get_table_index_from_float_format(x);
        z = polynomial_approximation(x, b);
        z = predicate ? z : -z;
        output[i] = z;
    }
}
