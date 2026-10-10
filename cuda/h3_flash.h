#pragma once
#include <cuda_runtime.h>

extern "C" cudaError_t h3_flash_bf16(void *output, const void *query, const void *key,
                        const void *value, float *logsumexp, unsigned batch,
                        unsigned sequence, unsigned heads, unsigned kv_heads,
                        float scale, bool causal, cudaStream_t stream);
