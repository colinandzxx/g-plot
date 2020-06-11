#include <stdio.h>
#include <unistd.h>
#include <string>
#include <time.h>
#include <gtest/gtest.h>

#include "g_plot.h"

TEST(shabal, 01)
{
    GPU_Count();
    printf("this is a test\n");
    clock_t start = clock();
    ASSERT_TRUE(testCuda() == 0);
    clock_t end = clock();
    printf("tm: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
}

TEST(shabal, plot_ctx)
{
    clock_t start = clock();
    void *buffer = malloc(10240ul * 8192 * 32);
    testPlot(buffer);
    clock_t end = clock();
    printf("tm: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    FILE *fp = fopen("/home/yuncun/ylh/remote_project/g-sphshabal-sample/build/test.bin", "wb");
    for (uint64_t i = 0; i < 8192 * 32; i += 64)
    {
        for (uint64_t j = 0; j < 8; j++)
        {
            fwrite(&((uint8_t *)buffer)[j * 8192 * 32 + i], 1, 64, fp);
        }
    }
    fclose(fp);
}

TEST(shabal, test_C_Process)
{
    test_C_Process();
}