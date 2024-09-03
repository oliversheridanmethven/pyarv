#include "version.h"

/* These I will read from git and cmake. */
char *repo_name(void) {
    return "testing";
}

char *repo_version(void) {
    return "v0.0.4-6-gd5bd028-dirty";
}


/* I could change the following to use git, but then contributors
 * might get listed also, which I don't want for now. */
char *repo_author(void) {
    return "Dr Oliver Sheridan-Methven";
}

char *repo_email(void) {
    return "oliver.sheridan-methven@hotmail.co.uk";
}
