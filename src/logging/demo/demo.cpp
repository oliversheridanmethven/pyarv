#include "logging/logging.hpp"

int main(int argc, char **argv) {
//    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_v = 0;
    FLAGS_logtostdout = true;
    google::InitGoogleLogging(argv[0]);

    LOG_INFO << "some info";
    LOG_DEBUG << "some debug";
    LOG_WARNING << "some warning";
}
