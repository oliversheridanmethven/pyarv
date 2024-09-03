#ifndef LOGGING_H_
#define LOGGING_H_

#include <iso646.h>// For nicer boolean operator spellings.
#include <signal.h>
#include <stdlib.h>

/* Some wrappers. */
#define LOG_INFO(...) printf(__VA_ARGS__)
#define LOG_MESSAGE(...) printf(__VA_ARGS__)
#define LOG_PRINT(...) printf(__VA_ARGS__)
#define LOG_PRINTERR(...) fprintf(stderr, __VA_ARGS__)
#define LOG_DEBUG(...) printf(__VA_ARGS__)
#define LOG_WARNING(...) fprintf(stderr, __VA_ARGS__)
#define LOGGING_ERROR_SIGNAL SIGTRAP
#define LOG_ERROR(...)            \
    fprintf(stderr, __VA_ARGS__); \
    raise(LOGGING_ERROR_SIGNAL)

#endif /* LOGGING_H_ */
