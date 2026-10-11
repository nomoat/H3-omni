#ifndef H3_VSA_HOPPER_H
#define H3_VSA_HOPPER_H
#include <cuda_runtime.h>
extern "C" cudaError_t h3_vsa_hopper(void *out, const void *q, const void *k, const void *v,
                                     float *lse, int *indices, int *counts, int *sizes,
                                     unsigned tiles, unsigned heads, cudaStream_t stream);
#endif
