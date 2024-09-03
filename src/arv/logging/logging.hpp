#ifndef TESTING_LOGGING_HPP
#define TESTING_LOGGING_HPP

#include <glog/logging.h> // The logging library of choice.

#define INFO_LEVEL 1
#define DEBUG_LEVEL 2
#define TRACE_LEVEL 3

/* Some wrappers. */
#define LOG_INFO VLOG(INFO_LEVEL)
#define LOG_DEBUG VLOG(DEBUG_LEVEL)
#define LOG_TRACE VLOG(TRACE_LEVEL)
#define LOG_WARNING LOG(WARNING)
#define LOG_ERROR LOG(ERROR)
#define LOG_CRITICAL LOG(FATAL)
#define LOG_FAILURE LOG(FATAL)

#define LOG_INIT(argv) google::InitGoogleLogging(argv[0]);

#endif //TESTING_LOGGING_HPP
