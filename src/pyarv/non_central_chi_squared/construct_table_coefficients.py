import pprint

table_size = 16
interpolation_functions = 16
polynomial_orders = 1
TABLE_MAX_INDEX = (table_size - 1)  # Zero indexing...


def compute_coefficient(*args, **kwargs):
    return None


polynomial_coefficients = {}
for half in [0, 1]:
    polynomial_coefficients[half] = {}
    for interpolation_function in range(interpolation_functions):
        polynomial_coefficients[half][interpolation_function] = {}
        for polynomial_order in range(polynomial_orders):
            polynomial_coefficients[half][interpolation_function][polynomial_order] = [None]*table_size
            for table_entry in range(table_size):
                f_lower = func
                f_upper = lambda u, *args, **kwargs: func(1.0 - u, *args, **kwargs)
                coeffs_lower = piecewise_polynomial_coefficients_in_half_interval(f_lower, n_intervals, polynomial_order)
                coeffs_upper = piecewise_polynomial_coefficients_in_half_interval(f_upper, n_intervals, polynomial_order)

    polynomial_coefficients[half][interpolation_function][polynomial_order][table_entry] = compute_coefficient(
                    half=half,
                    interpolation_function=interpolation_function,
                    polynomial_order=polynomial_order,
                    table_entry=table_entry)

pprint.pprint(polynomial_coefficients)
