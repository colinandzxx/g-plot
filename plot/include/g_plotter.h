#ifndef __G_PLOTTER__
#define __G_PLOTTER__

#include "stdint.h"

struct plotter
{
    uint8_t abort;

    uint64_t addr;
    uint64_t nonce_nr;
    uint64_t range;
    uint64_t plotted;
    uint64_t progress;

    uint32_t gpuid;

    void *plot_ctx;
};
typedef struct plotter Plotter;

typedef void (*PLOT_CALLBACK)(uint64_t progress, uint8_t *sols);

#ifdef __cplusplus
extern "C"
{
#endif

    void *create_plotter(uint64_t addr, uint64_t nonce_nr, uint64_t range, uint32_t gpuid);
    void destroy_plotter(void *plotter);
    void abort_plotter(void *plotter);
    void reset_plotter(void *plotter);
    void run_plotter(void *plotter, PLOT_CALLBACK callback);

#ifdef __cplusplus
}
#endif

#endif //__G_PLOTTER__