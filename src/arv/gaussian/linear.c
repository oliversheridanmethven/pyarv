#define POLYNOMIAL_ORDER 1
#define TABLE_SIZE 8

#include "linear.h"
#include "approximation/approximation.h"
#include "polynomial_coefficients_order_1_table_size_8.h"

#define TABLE_MAX_INDEX (TABLE_SIZE - 1)// Zero indexing...

#include <stdlib.h>
#if __STDC_VERSION__ < 202311L
#include <stdbool.h>
#endif

static inline Float polynomial_approximation(Float u, UInt b)
{
    return poly_order_1_table_size_8_coef_0[b] + u * poly_order_1_table_size_8_coef_1[b];
}

void linear(const Float *restrict const input,
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
