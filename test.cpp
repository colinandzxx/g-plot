#include <stdio.h>
#include <unistd.h>
#include <string>
#include <time.h>
#include "test.h"
// extern "C"
// {
// #include "sph_shabal.h"
// }

// extern "C" int testCuda();

int main()
{
    printf("this is a test\n");

    clock_t start = clock();
    testCuda();
    clock_t end = clock();
    printf("tm: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // sph_shabal512_context cc;
    // sph_shabal512_init(&cc);
    // std::string str = "this is a test";
    // sph_shabal512(&cc, str.c_str(), str.length());
    // char dst[32] = {};
    // sph_shabal512_close(&cc, dst);

    // printf("%s\n", dst);

    // sleep(5);

    return 1;
}