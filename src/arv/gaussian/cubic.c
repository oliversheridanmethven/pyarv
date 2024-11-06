#define POLYNOMIAL_ORDER 3
#define TABLE_SIZE 16

#include "cubic.h"
#include "approximation/approximation.h"
#include "polynomial_coefficients_order_3_table_size_16.h"

#define TABLE_MAX_INDEX (TABLE_SIZE - 1)// Zero indexing...

#include <stdlib.h>
#if __STDC_VERSION__ < 202311L
#include <stdbool.h>
#endif


static inline Float polynomial_approximation(Float u, UInt b)
{
    Float z, z_even, z_odd;
    //  return poly_coef_0[b] + u * (poly_coef_1[b] + u * (poly_coef_2[b] + u * poly_coef_3[b]));
    Float x = u * u;
    z_even = poly_order_3_table_size_16_coef_0[b] + poly_order_3_table_size_16_coef_2[b] * x;
    z_odd = poly_order_3_table_size_16_coef_1[b] + poly_order_3_table_size_16_coef_3[b] * x;
    z = z_even + z_odd * u;
    return z;
}

void cubic(const Float *restrict const input,
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
        b = cap_index(b, TABLE_MAX_INDEX);
        z = polynomial_approximation(x, b);
        z = predicate ? z : -z;
        output[i] = z;
    }
}
