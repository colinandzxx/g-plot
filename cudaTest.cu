#include "sph_shabal.h"
#include "test.h"
#include <cuda.h>
#include <stdio.h>

__global__ void foo() {}

int testCuda() {
  foo<<<1, 1>>>();
  printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

  // sph_shabal512_context cc;
  // sph_shabal512_init(&cc);
  // std::string str = "this is a test";
  // sph_shabal512(&cc, str.c_str(), str.length());
  // char dst[32] = {};
  // sph_shabal512_close(&cc, dst);

  return 0;
}