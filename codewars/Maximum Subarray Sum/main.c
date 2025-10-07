#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

short is_full_negative(const int* array, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (array[i] > 0) {
            return 0;
        }
    }
    return 1;
}

int maxSequence(const int* array, size_t n) {

    if (is_full_negative(array, n)) {
        return 0;
    }

    int curr = 0, max = curr;

    for (size_t i = 0; i < n; ++i) {

        curr += array[i];
        if (curr > max) {
            max = curr;
        }
        if (curr < 0) {
            curr = 0;
        }
    }

    return max;
}

int main() {

    FILE* input = freopen("input.txt", "r", stdin);
    
    size_t size;
    scanf("%zu\n", &size);

    int* arr = (int*)malloc(size);
    for (size_t i = 0 ; i < size; ++i) {
        scanf("%i", &arr[i]);
    }
    
    printf("%d", maxSequence(arr, size));

    fclose(input);

    return 0;
}