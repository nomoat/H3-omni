// Standalone adapter for the exact FlashAttention source revision in PyTorch 2.9.1.
// This is an ordinary attention operator dependency, like cuBLAS for GEMM.
// Dropout, backward, KV cache, local attention and softcap are not instantiated.
#include "h3_flash.h"
#define FLASH_NAMESPACE h3_phase1_flash
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace h3_flash_adapter {
using Traits = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false,
                                                       cutlass::bfloat16_t>;
template <bool Causal, bool Even>
__global__ void forward_kernel(const __grid_constant__ h3_phase1_flash::Flash_fwd_params p) {
    h3_phase1_flash::compute_attn<Traits, false, Causal, false, false, Even, true, false, false>(p);
}

template <bool Causal, bool Even>
cudaError_t launch(h3_phase1_flash::Flash_fwd_params &p, cudaStream_t stream) {
    auto kernel = forward_kernel<Causal, Even>;
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     Traits::kSmemSize);
    if (error != cudaSuccess) return error;
    dim3 grid((p.seqlen_q + 127) / 128, p.b, p.h);
    kernel<<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(p);
    return cudaGetLastError();
}
} // namespace h3_flash_adapter

cudaError_t h3_flash_bf16(void *output, const void *query, const void *key,
                        const void *value, float *logsumexp, unsigned batch,
                        unsigned sequence, unsigned heads, unsigned kv_heads,
                        float scale, bool causal, cudaStream_t stream) {
    using h3_flash_adapter::launch;
    h3_phase1_flash::Flash_fwd_params p{};
    p.q_ptr = const_cast<void *>(query);
    p.k_ptr = const_cast<void *>(key);
    p.v_ptr = const_cast<void *>(value);
    p.o_ptr = output;
    p.softmax_lse_ptr = logsumexp;
    p.q_row_stride = p.o_row_stride = heads * 128;
    p.k_row_stride = p.v_row_stride = kv_heads * 128;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.q_batch_stride = p.o_batch_stride = int64_t(sequence) * heads * 128;
    p.k_batch_stride = p.v_batch_stride = int64_t(sequence) * kv_heads * 128;
    p.b = batch;
    p.h = heads;
    p.h_k = kv_heads;
    p.h_h_k_ratio = heads / kv_heads;
    p.seqlen_q = p.seqlen_k = sequence;
    p.seqlen_q_rounded = p.seqlen_k_rounded = (sequence + 127) / 128 * 128;
    p.total_q = batch * sequence;
    p.d = p.d_rounded = 128;
    p.scale_softmax = p.scale_softmax_rp_dropout = scale;
    p.scale_softmax_log2 = scale * 1.4426950408889634074;
    p.p_dropout = p.rp_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.is_bf16 = true;
    p.is_causal = causal;
    p.is_seqlens_k_cumulative = true;
    p.window_size_left = causal ? int(sequence) : -1;
    p.window_size_right = causal ? 0 : -1;
    p.num_splits = 1;
    const bool even = sequence % 128 == 0;
    if (causal)
        return even ? launch<true, true>(p, stream) : launch<true, false>(p, stream);
    return even ? launch<false, true>(p, stream) : launch<false, false>(p, stream);
}
