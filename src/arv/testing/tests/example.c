/* An example of using the Criterion testing framework. */

#include "testing/testing.h"
#include <string.h>

/* This is what a failing test looks like. */
Test(example, test_failure, .disabled = true)
{
    cr_expect(eq(int, 1, 2), "This should fail.");
}

Test(example, test_expect)
{
    cr_expect(strlen("Hello") == 5, "This should pass.");
}

Test(example, test_fail)
{
    cr_assert(eq(int, 2, 1 + 1), "This should pass.");
    cr_assert(not(eq(int, 1, 2)), "This should pass.");
}

Test(logging, test_exit, .exit_code = 1)
{
    exit(1);
}

Test(capturing_output, std_output, .init = redirect_all_stdout, .disabled = true)
{
    /* Issue reported: https://github.com/Snaipe/Criterion/issues/508 */
    fprintf(stderr, "something");
    cr_assert_stderr_eq_str("something");
    cr_assert_stdout_neq_str("something else");
    fprintf(stderr, " else");
    cr_assert_stdout_eq_str("something else");
    cr_assert_stdout_eq_str("something\n");
    fprintf(stderr, "error");
    cr_assert_stderr_eq_str("error");
    cr_assert_stderr_neq_str("another");
}

Test(capturing_output, std_output_failure, .init = redirect_all_stdout, .disabled = true)
{
    fprintf(stderr, "something");
    cr_assert_stderr_eq_str("something else");
}
