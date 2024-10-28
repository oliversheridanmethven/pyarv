#include "linear.h"
#include "approximation/approximation.h"
#include <math.h>

#define TABLE_SIZE 16
#define INTERPOLATION_FUNCTIONS 16
#define HALVES 2
#define POLYNOMIAL_ORDER 1
#define TABLE_MAX_INDEX (TABLE_SIZE - 1)// Zero indexing...

typedef Float Polynomial_Coefficients[HALVES][INTERPOLATION_FUNCTIONS][POLYNOMIAL_ORDER + 1][TABLE_SIZE];

#pragma omp declare simd
static inline void interpolation_indices(const Float y,
                                         UInt *restrict interpolation_index_lower,
                                         UInt *restrict interpolation_index_upper,
                                         Float *restrict weight_lower,
                                         Float *restrict weight_upper)
{
    Float x = sqrtf(y) * (INTERPOLATION_FUNCTIONS - 1);
    *interpolation_index_lower = (UInt) x;
    *interpolation_index_upper = *interpolation_index_lower + 1;
    *weight_upper = x - ((UInt) x);
    *weight_lower = 1.0f - *weight_upper;
}

#pragma omp declare simd
static inline Float polynomial_linear_approximation(Float u,
                                                    UInt b,
                                                    UInt h,
                                                    UInt i,
                                                    const Polynomial_Coefficients polynomial_coefficients)
{
    /*
     * Polynomial approximation of a function.
     *
     * Input:
     *      u - Input position.
     *      b - Index of polynomial coefficient to use.
     *      h - Index of which half to use.
     *      i - Index of which interpolating function to use.
     *
     */
    Float poly_coef_0 = polynomial_coefficients[h][i][0][b];
    Float poly_coef_1 = polynomial_coefficients[h][i][1][b];
    Float z = poly_coef_0 + poly_coef_1 * u;
    return z;
}

void linear(const Float *restrict const input,
            Float *restrict const output,
            const size_t input_buffer_size,
            const Float *restrict const non_centrality,
            const Float degrees_of_freedom,
            const Float *restrict const polynomial_coefficients)
{
    // #pragma omp simd
    for (unsigned int i = 0; i < input_buffer_size; i++)
    {
        Float u, p, z, lambda;
        u = input[i];
        lambda = non_centrality[i];
        UInt upper_half = u > 0.5f;
        u = upper_half ? 1.0f - u : u;
        Float weight_lower, weight_upper;
        UInt interpolation_index_lower, interpolation_index_upper;
        Float y = degrees_of_freedom / (lambda + degrees_of_freedom);// interpolation_value
        interpolation_indices(y, &interpolation_index_lower, &interpolation_index_upper, &weight_lower, &weight_upper);
        UInt b = get_table_index_from_float_format(u);
        b = cap_index(b, TABLE_MAX_INDEX);
        Float p_lower = polynomial_linear_approximation(u, b, upper_half, interpolation_index_lower, polynomial_coefficients);
        Float p_upper = polynomial_linear_approximation(u, b, upper_half, interpolation_index_upper, polynomial_coefficients);
        p = weight_lower * p_lower + weight_upper * p_upper;
        z = lambda + degrees_of_freedom + 2.0f * sqrtf(lambda + degrees_of_freedom) * p;
        output[i] = z;
    }
}
