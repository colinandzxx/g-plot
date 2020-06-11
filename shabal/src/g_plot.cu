#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "sph_shabal.cuh"
#include "sph_swap.h"
#include "g_plot.h"

// https://zh.wikipedia.org/wiki/NVIDIA%E9%A1%AF%E7%A4%BA%E6%A0%B8%E5%BF%83%E5%88%97%E8%A1%A8#GeForce_900
// GTX1050 Ti 4G / GTX1060 6G
#define HASH_SIZE 32ul
#define HASHES_PER_SCOOP 2ul
#define SCOOP_SIZE (HASH_SIZE * HASHES_PER_SCOOP)
#define SCOOPS_PER_NONCE 4096ul
#define NONCE_SIZE (SCOOP_SIZE * SCOOPS_PER_NONCE)
#define ADDR_SIZE 8ul

#define NONCE_NR_SIZE 8ul
#define BASE_SIZE (ADDR_SIZE + NONCE_NR_SIZE)
#define PLOT_SIZE (NONCE_SIZE + BASE_SIZE)
#define HASH_CAP 4096ul

#define THREADS_PER_BLOCK 512ul
#define GIRD_NUM 10ul

#define MAX_NAME_LEN 256

// last error reason, to be picked up by stats
// to be returned to caller
char LAST_ERROR_REASON[MAX_NAME_LEN];

#define checkCudaErrors_V(ans)                               \
  ({                                                         \
    if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) \
      return;                                                \
  })
#define checkCudaErrors_N(ans)                               \
  ({                                                         \
    if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) \
      return NULL;                                           \
  })
#define checkCudaErrors(ans)                           \
  ({                                                   \
    int retval = gpuAssert((ans), __FILE__, __LINE__); \
    if (retval != cudaSuccess)                         \
      return retval;                                   \
  })

inline int gpuAssert(cudaError_t code, const char *file, int line,
                     bool abort = true)
{
  int device_id;
  cudaGetDevice(&device_id);
  if (code != cudaSuccess)
  {
    snprintf(LAST_ERROR_REASON, MAX_NAME_LEN, "Device %d GPUassert: %s %s %d",
             device_id, cudaGetErrorString(code), file, line);
    cudaDeviceReset();
    if (abort)
      return code;
  }
  return code;
}

__global__ void testSha(uint8_t *g_out)
{
  uint32_t global_id = blockDim.x * blockIdx.x + threadIdx.x;
  // uint32_t gid = global_id / THREADS_PER_BLOCK;
  // uint32_t bid = global_id % THREADS_PER_BLOCK;

  for (int i = 0; i < SCOOPS_PER_NONCE * 2; i++)
  {
    sph_shabal256_context cc;
    sph_shabal256_init(&cc);
    char str[] =
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "2312312312312312312312312312312312312312312312312312312312312312312312"
        "3123123123123123123123123123123123123123123123123123123123123123123123"
        "1231231231231231231231231231231231231231231231231231231231231231231231"
        "231231231231231231231231231231231231";
    sph_shabal256(&cc, str, sizeof(str) - 1);
    // memcpy(g_out, "123", 3);
    // sph_shabal256(&cc, g_out, 3);
    // char dst[32] = {};
    sph_shabal256_close(&cc, &g_out[global_id * HASH_SIZE]);
    // sph_shabal256_close(&cc, dst);
    // printf("AAAA: %s\n", dst);
  }
}

int testCuda()
{
  uint8_t *out;
  uint8_t *g_out;

  out = (uint8_t *)malloc(GIRD_NUM * THREADS_PER_BLOCK * NONCE_SIZE);
  if (out == NULL)
  {
    printf("E00: malloc err for nonce\n");
  }

  if (cudaMalloc((void **)&g_out, GIRD_NUM * THREADS_PER_BLOCK * NONCE_SIZE) !=
      cudaSuccess)
  {
    printf("E01: cuda alloc memory error for nonce (%s)\n",
           cudaGetErrorString(cudaGetLastError()));
    return 1;
  }

  testSha<<<GIRD_NUM, THREADS_PER_BLOCK>>>(g_out);
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("E02: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaDeviceSynchronize();

  if (cudaMemcpy(out, g_out, GIRD_NUM * THREADS_PER_BLOCK * NONCE_SIZE,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("E03: cuda memory copy error for nonce (%s)\n",
           cudaGetErrorString(cudaGetLastError()));
    return 1;
  }

  cudaFree(g_out);

  // for (int i = 0; i < GIRD_NUM * THREADS_PER_BLOCK * 32; i += 32) {
  //   for (int j = 0; j < 32; j++) {
  //     printf("%02x", out[i + j]);
  //   }
  //   printf("\n");
  // }

  cudaDeviceReset();
  return 0;
}

__global__ void plot_poc1(uint64_t *addr, uint64_t *start_nr, uint8_t *g_sols)
{
  uint32_t global_id = blockDim.x * blockIdx.x + threadIdx.x;
  // uint32_t gid = global_id / THREADS_PER_BLOCK;
  // uint32_t bid = global_id % THREADS_PER_BLOCK;

  uint64_t nonce_nr = *start_nr + global_id;
  uint8_t *sol = &g_sols[global_id * PLOT_SIZE];
  uint64_t swap_addr = SPH_SWAP64(*addr);
  memcpy(&sol[NONCE_SIZE], &swap_addr, 8);
  uint64_t swap_nonce_nr = SPH_SWAP64(nonce_nr);
  memcpy(&sol[NONCE_SIZE + ADDR_SIZE], &swap_nonce_nr, 8);

  sph_shabal256_context cc;
  for (int i = NONCE_SIZE; i > 0; i -= HASH_SIZE)
  {
    sph_shabal256_init(&cc);
    auto len = PLOT_SIZE - i;
    if (len > HASH_CAP)
    {
      len = HASH_CAP;
    }
    sph_shabal256(&cc, &sol[i], len);
    sph_shabal256_close(&cc, &sol[i - HASH_SIZE]);
  }
  uint8_t final[HASH_SIZE];
  sph_shabal256_init(&cc);
  sph_shabal256(&cc, sol, PLOT_SIZE);
  sph_shabal256_close(&cc, final);

  for (int i = 0; i < NONCE_SIZE; i++)
  {
    sol[i] = sol[i] ^ final[i % HASH_SIZE];
  }
}

struct plotter_ctx
{
  uint64_t grids;
  uint8_t *sols;

  uint64_t *g_addr;
  uint64_t *g_start_nr;
  uint8_t *g_sols;

  plotter_ctx(const uint64_t grids) : grids(grids)
  {
    sols = (uint8_t *)malloc(PLOT_SIZE * THREADS_PER_BLOCK * grids);
    assert(sols != NULL);

    checkCudaErrors_V(cudaMalloc((void **)&g_addr, sizeof(uint64_t)));
    checkCudaErrors_V(cudaMalloc((void **)&g_start_nr, sizeof(uint64_t)));
    checkCudaErrors_V(
        cudaMalloc((void **)&g_sols, PLOT_SIZE * THREADS_PER_BLOCK * grids));
  }

  ~plotter_ctx()
  {
    free(sols);
    checkCudaErrors_V(cudaFree(g_addr));
    checkCudaErrors_V(cudaFree(g_start_nr));
    checkCudaErrors_V(cudaFree(g_sols));
    cudaDeviceReset();
  }

  void reset()
  {
    // free(sols);
    bzero(sols, PLOT_SIZE * THREADS_PER_BLOCK * grids);
    checkCudaErrors_V(cudaFree(g_addr));
    checkCudaErrors_V(cudaFree(g_start_nr));
    checkCudaErrors_V(cudaFree(g_sols));
    cudaDeviceReset();

    sols = (uint8_t *)malloc(PLOT_SIZE * THREADS_PER_BLOCK * grids);
    assert(sols != NULL);
    checkCudaErrors_V(cudaMalloc((void **)&g_addr, sizeof(uint64_t)));
    checkCudaErrors_V(cudaMalloc((void **)&g_start_nr, sizeof(uint64_t)));
    checkCudaErrors_V(
        cudaMalloc((void **)&g_sols, PLOT_SIZE * THREADS_PER_BLOCK * grids));
  }

  int plot(uint64_t addr, uint64_t start_nr, uint64_t &num)
  {
    num = 0;
    checkCudaErrors(
        cudaMemcpy(g_addr, &addr, sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(g_start_nr, &start_nr, sizeof(uint64_t),
                               cudaMemcpyHostToDevice));

    plot_poc1<<<grids, THREADS_PER_BLOCK>>>(g_addr, g_start_nr, g_sols);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(sols, g_sols,
                               PLOT_SIZE * THREADS_PER_BLOCK * grids,
                               cudaMemcpyDeviceToHost));
    num = grids * THREADS_PER_BLOCK;
    return 0;
  }
};

void GPU_Count()
{
  int num;
  cudaDeviceProp prop;
  cudaGetDeviceCount(&num);
  printf("deviceCount := %d\n", num);
  for (int i = 0; i < num; i++)
  {
    cudaGetDeviceProperties(&prop, i);
    printf("name:%s\n", prop.name);
    printf("totalGlobalMem:%lu GB\n", prop.totalGlobalMem / 1024 / 1024 / 1024);
    printf("multiProcessorCount:%d\n", prop.multiProcessorCount);
    printf("maxThreadsPerBlock:%d\n", prop.maxThreadsPerBlock);
    printf("sharedMemPerBlock:%lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("major:%d,minor:%d\n", prop.major, prop.minor);
  }
}

void testPlot(void *buffer)
{
  plotter_ctx *ctx = new plotter_ctx(10);
  ctx->reset();
  uint64_t num = 0;
  auto ret = ctx->plot(1ul, 0ul, num);
  if (ret != 0)
  {
    printf("last err: %s\n", LAST_ERROR_REASON);
  }
  if (buffer)
  {
    for (uint64_t i = 0; i < num; i++)
    {
      memcpy(&((uint8_t *)buffer)[NONCE_SIZE * i], &ctx->sols[PLOT_SIZE * i],
             NONCE_SIZE);
      ;
    }

    // for (uint64_t i = 0; i < NONCE_SIZE * num; i += NONCE_SIZE) {
    //   for (int j = 0; j < HASH_SIZE; j++) {
    //     printf("%02x", ((uint8_t *)buffer)[i + j]);
    //   }
    //   printf("\n");
    // }
  }
  free(ctx);
}