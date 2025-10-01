#define _POSIX_C_SOURCE 200809L 
#include "stdio.h"
#include "string.h"

char *DuplicateEncoder(const char *string)
{
    short num_of_symbols[256] = {0};
    char* result = strdup(string);

    for (int i = 0; i < strlen(result); ++i) {
        if (result[i] < 91 && result[i] >= 65) {
            result[i] += 32;
        }
    }

    for (size_t i = 0; i < strlen(result); ++i) {
        num_of_symbols[result[i]] += 1;
    }

    for (int i = 0; i < strlen(result); ++i) {
        if (num_of_symbols[result[i]] > 1) {
            result[i] = ')';
        } else {
            result[i] = '(';
        }
    }    

    return result;
}

int main() {
    const char* solution = DuplicateEncoder("  hui   ");
    printf("%s", solution);
    return 0;
}                                                                                            