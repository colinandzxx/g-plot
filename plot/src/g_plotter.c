#include <stdio.h>
#include <assert.h>
#include "g_plot.h"
#include "g_plotter.h"

#define MIN_GRIDS 5ul

void test_C_Process()
{
    void *ctx = create_plot_ctx(1, 0);
    uint64_t num = 0;
    run_plot_ctx(ctx, 1, 0, &num);
    destroy_plot_ctx(ctx);
}

void *create_plotter(uint64_t addr, uint64_t nonce_nr, uint64_t range, uint32_t gpuid)
{
    Plotter *p = (Plotter *)malloc(sizeof(Plotter));
    p->abort = 0;
    p->addr = addr;
    p->nonce_nr = nonce_nr;
    p->range = range;
    p->plotted = 0;
    p->progress = 0;
    p->gpuid = gpuid;
    p->plot_ctx = create_plot_ctx(MIN_GRIDS, gpuid);
    return p;
}

void destroy_plotter(void *plotter)
{
    if (plotter)
    {
        destroy_plot_ctx(((Plotter *)plotter)->plot_ctx);
        free(plotter);
    }
}

void abort_plotter(void *plotter)
{
    ((Plotter *)plotter)->abort = 1;
}

void reset_plotter(void *plotter)
{
    Plotter *p = ((Plotter *)plotter);
    if (p && p->plot_ctx)
    {
        ((Plotter *)plotter)->abort = 0;
        reset_plot_ctx(p->plot_ctx);
    }
}

void run_plotter(void *plotter, PLOT_CALLBACK callback)
{
    Plotter *p = ((Plotter *)plotter);
    if (p == NULL)
    {
        return;
    }

    while (p->abort == 0)
    {
        uint64_t num = 0;
        int ret = run_plot_ctx(p->plot_ctx, p->addr, p->nonce_nr, &num);
        assert(ret == 0);
        p->plotted += num;
        if (p->plotted > p->range)
        {
            p->plotted = p->range;
            p->abort = 1;
        }
        p->progress = p->plotted * 100 / p->range;
        callback(p->progress, get_plot_ctx_sols(p->plot_ctx));
    }
}