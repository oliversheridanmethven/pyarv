#include <stdbool.h>

void foo(const float *restrict const input,
float *restrict const output,
const int input_buffer_size)
{
    #pragma omp simd
    for (int i = 0; i < input_buffer_size; i++)
    {
        float x, z;
//        x = input[i]; 
//        bool predicate;
        z = (1.0 < x) ? z : -z;
        output[i] = z;
    }
}
