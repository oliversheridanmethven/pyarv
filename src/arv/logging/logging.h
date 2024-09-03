#ifndef LOGGING_H_
#define LOGGING_H_

#include <glib.h>  // The logging library of choice.
#include <iso646.h>// For nicer boolean operator spellings.
#include <signal.h>
#include <stdlib.h>

/* Some wrappers. */
#define LOG_INFO(...) g_info(__VA_ARGS__)
#define LOG_MESSAGE(...) g_message(__VA_ARGS__)
#define LOG_PRINT(...) g_print(__VA_ARGS__)
#define LOG_PRINTERR(...) g_printerr(__VA_ARGS__)
#define LOG_DEBUG(...) g_debug(__VA_ARGS__)
#define LOG_WARNING(...) g_warning(__VA_ARGS__)
#define LOG_ERROR(...) g_error(__VA_ARGS__)

#define LOGGING_ERROR_SIGNAL SIGTRAP

/* g_error raises this via G_BREAKPOINT
 * cf. https://docs.gtk.org/glib/func.BREAKPOINT.html
 * */

void show_all_logging(void)
{
    if (setenv("G_MESSAGES_DEBUG", "all", 1))
    {
        LOG_ERROR("Unable to set the G_MESSAGES_DEBUG environment variable.");
    };
}

#endif /* LOGGING_H_ */
