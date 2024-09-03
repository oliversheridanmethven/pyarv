#include "logging/logging.hpp"
#include "testing/testing.h"

TEST(logging, minimal_strings) {
    LOG_INFO << "some info";
    LOG_DEBUG << "some debug";
    LOG_WARNING << "some warning";
}

TEST(logging, variable_args) {
    LOG_INFO << "some info" << "some string" << 10;
}

TEST(logging, error_fails) {
    LOG_ERROR << "some error";// This terminates the program.
}
