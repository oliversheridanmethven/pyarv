#include "error_codes.h"
#include <stdio.h>

char *error_message;

void set_error_message(char *message) {
    error_message = message;
}

void unset_error_message(void) {
    error_message = nullptr;
}

char *get_error_message(void) {
    return error_message;
}

void print_error_message(void) {
    fprintf(stderr, "%s", error_message);
}
