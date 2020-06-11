#include "stdio.h"
#include "g_plot.h"

void test_C_Process()
{
    void *ctx = create_plot_ctx(1, 0);
    uint64_t num = 0;
    run_plot(ctx, 1, 0, &num);
    destroy_plot_ctx(ctx);
}
