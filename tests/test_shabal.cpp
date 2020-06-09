#include <stdio.h>
#include <unistd.h>
#include <string>
#include <time.h>
#include "test_shabal.h"

#include <gtest/gtest.h>

// int main()
// {
//     printf("this is a test\n");

//     clock_t start = clock();
//     testCuda();
//     clock_t end = clock();
//     printf("tm: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

//     return 0;
// }

TEST(shabal, 01)
{
    printf("this is a test\n");
    clock_t start = clock();
    testCuda();
    clock_t end = clock();
    printf("tm: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
}