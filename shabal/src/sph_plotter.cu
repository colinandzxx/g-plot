#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "sph_shabal.cuh"

// https://zh.wikipedia.org/wiki/NVIDIA%E9%A1%AF%E7%A4%BA%E6%A0%B8%E5%BF%83%E5%88%97%E8%A1%A8#GeForce_900
// GTX960 GTX1060
#define HASH_SIZE 32
#define HASHES_PER_SCOOP 2
#define SCOOP_SIZE (HASH_SIZE * HASHES_PER_SCOOP)
#define SCOOPS_PER_NONCE 4096
#define NONCE_SIZE (SCOOP_SIZE * SCOOPS_PER_NONCE)
#define HASH_CAP 4096

#define THREADS_PER_BLOCK 1024
#define GIRD_NUM 10240

#define MAX_NAME_LEN 256

// last error reason, to be picked up by stats
// to be returned to caller
char LAST_ERROR_REASON[MAX_NAME_LEN];

#define checkCudaErrors_V(ans)                                                 \
  ({                                                                           \
    if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess)                   \
      return;                                                                  \
  })
#define checkCudaErrors_N(ans)                                                 \
  ({                                                                           \
    if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess)                   \
      return NULL;                                                             \
  })
#define checkCudaErrors(ans)                                                   \
  ({                                                                           \
    int retval = gpuAssert((ans), __FILE__, __LINE__);                         \
    if (retval != cudaSuccess)                                                 \
      return retval;                                                           \
  })

inline int gpuAssert(cudaError_t code, const char *file, int line,
                     bool abort = true) {
  int device_id;
  cudaGetDevice(&device_id);
  if (code != cudaSuccess) {
    snprintf(LAST_ERROR_REASON, MAX_NAME_LEN, "Device %d GPUassert: %s %s %d",
             device_id, cudaGetErrorString(code), file, line);
    cudaDeviceReset();
    if (abort)
      return code;
  }
  return code;
}

#define FOR_TEST 1
__global__ void sha(uint8_t *g_out) {
  uint32_t global_id = blockDim.x * blockIdx.x + threadIdx.x;
  // uint32_t id = global_id % THREADS_PER_BLOCK;

  // for (int i = 0; i < FOR_TEST; i++) {
    sph_shabal256_context cc;
    sph_shabal256_init(&cc);
    char str[] = "123";
    sph_shabal256(&cc, str, sizeof(str) - 1);
    // char dst[32] = {};
    sph_shabal256_close(&cc, g_out + global_id * 32);
    // printf("AAAA: %s\n", dst);
  // }
}

int testCuda() {
  uint8_t *g_out;
  uint8_t *out;

  out = (uint8_t *)malloc(GIRD_NUM * THREADS_PER_BLOCK * 32 * FOR_TEST);

  if (cudaMalloc((void **)&g_out, GIRD_NUM * THREADS_PER_BLOCK * 32 * FOR_TEST) !=
      cudaSuccess) {
    printf("E01: cuda alloc memory error for nonce\n");
    return 1;
  }

  sha<<<GIRD_NUM, THREADS_PER_BLOCK>>>(g_out);
  auto err = cudaGetLastError();
  printf("CUDA error: %s\n", cudaGetErrorString(err));
  if (err != cudaSuccess)
    return err;

  cudaDeviceSynchronize();

  if (cudaMemcpy(out, g_out, GIRD_NUM * THREADS_PER_BLOCK * 32 * FOR_TEST,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    printf("E07: copy memory error for g_nonce_id\n");
    return 1;
  }

  cudaFree(g_out);

  // for (int i = 0; i < GIRD_NUM * THREADS_PER_BLOCK * 32; i += 32)
  // {
  //   for (int j = 0; j < 32; j++)
  //   {
  //     printf("%02x", out[i + j]);
  //   }
  //   printf("\n");
  // }

  return 0;
}

__global__ void p_shabal(uint64_t start_nr, uint8_t *g_sols) {}

struct plotter_ctx {
  bool abort;
  uint32_t grids;
  uint8_t *sols;

  uint64_t *g_start_nr;
  uint8_t *g_sols;

  plotter_ctx(const uint32_t grids, const uint64_t nr)
      : abort(false), grids(grids) {
    sols = (uint8_t *)malloc(NONCE_SIZE * THREADS_PER_BLOCK * grids);
    assert(sols != NULL);

    checkCudaErrors_V(
        cudaMalloc((void **)&g_sols, NONCE_SIZE * THREADS_PER_BLOCK * grids));
  }

  ~plotter_ctx() {
    free(sols);
    checkCudaErrors_V(cudaFree(g_start_nr));
    checkCudaErrors_V(cudaFree(g_sols));
    cudaDeviceReset();
  }

  void reset() {
    cudaDeviceReset();
    abort = false;
    sols = (uint8_t *)malloc(NONCE_SIZE * THREADS_PER_BLOCK * grids);
    assert(sols != NULL);

    checkCudaErrors_V(
        cudaMalloc((void **)&g_sols, NONCE_SIZE * THREADS_PER_BLOCK * grids));
  }

  int plot(uint64_t start_nr) {
    checkCudaErrors(cudaMalloc((void **)&g_start_nr, sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(g_start_nr, &start_nr, sizeof(uint64_t),
                               cudaMemcpyHostToDevice));

    p_shabal<<<grids, THREADS_PER_BLOCK>>>(*g_start_nr, g_sols);

    return 0;
  }

  void plot_abort() { abort = true; }
};