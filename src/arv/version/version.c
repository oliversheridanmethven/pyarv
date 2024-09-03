#include "version.h"

/* These I will read from git and cmake. */
char *repo_name(void) {
    return "pyarv";
}

char *repo_version(void) {
    return "v0.0.1-1-gbb7a8c1-dirty";
}


/* I could change the following to use git, but then contributors
 * might get listed also, which I don't want for now. */
char *repo_author(void) {
    return "Dr Oliver Sheridan-Methven";
}

char *repo_email(void) {
    return "oliver.sheridan-methven@hotmail.co.uk";
}
