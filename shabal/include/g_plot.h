#ifndef __G_PLOT__
#define __G_PLOT__

extern "C"
{
    int testCuda();
    void GPU_Count();
    void testPlot(void *buffer);
}

#endif //__G_PLOT__