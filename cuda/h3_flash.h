#pragma once
#include <cuda_runtime.h>

extern "C" cudaError_t h3_flash_bf16(void *output, const void *query, const void *key,
                        const void *value, float *logsumexp, unsigned batch,
                        unsigned sequence, unsigned heads, unsigned kv_heads,
                        float scale, bool causal, cudaStream_t stream);
// FlashAttention-3 Hopper forward (cuda/flash3_attention.cu), non-causal.
extern "C" cudaError_t h3_flash3_bf16(void *output, const void *query, const void *key,
                        const void *value, float *logsumexp, unsigned batch,
                        unsigned sequence, unsigned heads, unsigned kv_heads,
                        float scale, cudaStream_t stream);
// The same kernel in FP16 with head dimension 64 (video VAE FP16 mode).
extern "C" cudaError_t h3_flash3_f16_d64(void *output, const void *query, const void *key,
                        const void *value, float *logsumexp, unsigned batch,
                        unsigned sequence, unsigned heads, float scale,
                        cudaStream_t stream);
