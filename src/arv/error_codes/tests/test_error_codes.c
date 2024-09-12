#include "error_codes/error_codes.h"
#include "testing/testing.h"

Test(error_codes, error_code_values_boolean)
{
    /*
     * This means we can use this easily in condition
     * statements to check if an error has occurred, and
     * follows the same style for general UNIX return
     * codes.
     * */
    cr_assert(EC_FAILURE);
    cr_assert_not(EC_SUCCESS);
}

Test(error_messages, message_starts_null)
{
    cr_expect_null(get_error_message());
}

Test(error_messages, set_not_null)
{
    set_error_message("Something");
    cr_expect_not_null(get_error_message());
}

Test(error_messages, set_unset)
{
    set_error_message("Something");
    unset_error_message();
    cr_expect_null(get_error_message());
}

Test(error_messages, printed_message, .init = redirect_all_stdout)
{
#define FIRST "Something"
#define SECOND "Something else"
    set_error_message(FIRST);
    print_error_message();
    set_error_message(SECOND);
    cr_assert_stderr_eq_str(FIRST);
    cr_assert_stderr_neq_str(SECOND);
}
