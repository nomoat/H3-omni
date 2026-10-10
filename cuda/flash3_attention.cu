// Standalone adapter for the FlashAttention-3 Hopper forward kernel
// (flash-attention v2.8.3, hopper/). BF16, head dimension 128, dense
// [batch, sequence, heads, 128] tensors, no dropout, split-KV, paging,
// softcap, local windows or variable lengths. Non-causal only: the DiT's
// joint attention. Causal text attention keeps the FlashAttention-2 adapter.
#include "h3_flash.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(
    Flash_fwd_params &params, cudaStream_t stream);

cudaError_t h3_flash3_bf16(void *output, const void *query, const void *key, const void *value,
                           float *logsumexp, unsigned batch, unsigned sequence, unsigned heads,
                           unsigned kv_heads, float scale, cudaStream_t stream) {
    static int sm_count = 0;
    if (!sm_count) {
        int device = 0;
        cudaError_t error = cudaGetDevice(&device);
        if (error == cudaSuccess)
            error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        if (error != cudaSuccess) return error;
    }
    if (heads % kv_heads) return cudaErrorInvalidValue;
    Flash_fwd_params p{};
    p.is_bf16 = true;
    p.q_ptr = const_cast<void *>(query);
    p.k_ptr = const_cast<void *>(key);
    p.v_ptr = const_cast<void *>(value);
    p.o_ptr = output;
    p.softmax_lse_ptr = logsumexp;
    p.q_row_stride = p.o_row_stride = int64_t(heads) * 128;
    p.k_row_stride = p.v_row_stride = int64_t(kv_heads) * 128;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.v_dim_stride = 1;
    p.q_batch_stride = p.o_batch_stride = int64_t(sequence) * heads * 128;
    p.k_batch_stride = p.v_batch_stride = int64_t(sequence) * kv_heads * 128;
    p.b = p.b_k = int(batch);
    p.h = int(heads);
    p.h_k = int(kv_heads);
    p.seqlen_q = p.seqlen_k = int(sequence);
    p.seqlen_q_rounded = p.seqlen_k_rounded = int((sequence + 127) / 128 * 128);
    p.total_q = p.total_k = int(batch * sequence);
    p.d = p.d_rounded = p.dv = p.dv_rounded = 128;
    p.scale_softmax = scale;
    p.softcap = 0.0f;
    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.is_causal = false;
    p.is_local = false;
    p.window_size_left = int(sequence) - 1;
    p.window_size_right = int(sequence) - 1;
    p.attention_chunk = 0;
    p.page_size = 1;
    p.num_splits = 1;
    p.pack_gqa = false;
    p.arch = 90;
    p.num_sm = sm_count;
    run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(p, stream);
    return cudaGetLastError();
}
