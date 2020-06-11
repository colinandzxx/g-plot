#ifndef __G_PLOT__
#define __G_PLOT__

#include "stdint.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int testCuda();
    void testPlot(void *buffer);
    void test_C_Process();

    void GPU_Count();

    void *create_plot_ctx(uint64_t grids, uint32_t gpuid);
    void destroy_plot_ctx(void *ctx);
    void reset_plot(void *ctx);
    int run_plot(void *ctx, uint64_t addr, uint64_t start_nr, uint64_t *num);

#ifdef __cplusplus
}
#endif

#endif //__G_PLOT__