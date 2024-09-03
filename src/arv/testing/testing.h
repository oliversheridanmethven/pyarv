#ifndef TESTING_HPP_
#define TESTING_HPP_

#ifdef __cplusplus

#include <gtest/gtest.h>
#include <gmock/gmock.h>

/* Taken from: https://stackoverflow.com/a/58369622/5134817. */
class CaptureStdOut : public ::testing::Test {
protected:
    CaptureStdOut() : cout_buffer{nullptr} {}

    ~CaptureStdOut() override = default;

    void SetUp() override {
        // Save cout's buffer...
        cout_buffer = std::cout.rdbuf();
        // Redirect cout to our stringstream buffer or any other ostream
        std::cout.rdbuf(captured_cout.rdbuf());
    }

    void TearDown() override {
        // When done redirect cout to its old self
        std::cout.rdbuf(cout_buffer);
        cout_buffer = nullptr;
    }

    std::stringstream captured_cout{};
    std::streambuf *cout_buffer;
};

#else

#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <criterion/redirect.h>

void redirect_all_stdout(void) {
    cr_redirect_stdout();
    setbuf(stdout, NULL);
    cr_redirect_stderr();
    setbuf(stderr, NULL);
}

#endif

#endif /*TESTING_HPP_*/
