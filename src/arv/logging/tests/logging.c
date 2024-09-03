#include "arv/logging/logging.h"
#include "testing/testing.h"

Test(logging, minimal_strings, .init = show_all_logging)
{
    LOG_INFO("some info");
    LOG_MESSAGE("some message");
    LOG_PRINT("some print\n");
    LOG_PRINTERR("some print error\n");
    LOG_DEBUG("some debug");
    LOG_WARNING("some warning");
}

Test(logging, variable_args)
{
    LOG_PRINT("some info, %s %i\n", "some string", 10);
}

Test(logging, error_fails, .signal = LOGGING_ERROR_SIGNAL)
{
    LOG_ERROR("some error");// This terminates the program.
}
