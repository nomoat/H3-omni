// Adapted from FastVideo d5253fd71055927f12aa6c4aa679b93a00238851.
// Forward kernel only, unchanged arithmetic; Apache-2.0 (FASTVIDEO-LICENSE).
// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>



using namespace kittens;
namespace cg = cooperative_groups;

// ThunderKittens kernels below use Hopper `wgmma`/TMA (sm_90a). In the multi-arch
// cu130 wheel they are also fed the sm_120a device pass, where ptxas rejects those
// instructions, so compile the kernel bodies only on sm_90a and emit empty stubs
// for every other device pass. __CUDA_ARCH_FEAT_SM90_ALL is defined only when the
// device pass targets sm_90a (same flag CUTLASS uses); non-Hopper GPUs pick a
// non-TK attention backend at runtime.
#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define FASTVIDEO_TK_HOPPER 1  // host pass or sm_90a device pass -> real body
#else
#define FASTVIDEO_TK_HOPPER 0  // any non-sm_90a device pass -> empty stub
#endif

constexpr int BLOCK_M             = 64;
constexpr int BLOCK_N             = 64;
template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (4*16);
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (4*16);
};
template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N;
    const int hr;
    const int max_kv_blocks_per_q;

    int32_t *__restrict__  q2k_block_sparse_index;
    int32_t *__restrict__  q2k_block_sparse_num;
    int32_t *__restrict__  block_size;
};


template<int D>
__global__  __launch_bounds__(128, 4)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) { // use block size of 64
#if FASTVIDEO_TK_HOPPER
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<64, K::tile_width>;
    using k_tile    =         st_bf<64, K::tile_width>;
    using v_tile    =         st_bf<64, K::tile_width>;
    using l_col_vec = col_vec<st_fl<64, K::tile_width>>;
    using o_tile    =         st_bf<64, K::tile_width>;

    q_tile    (&q_smem)[1] = al.allocate<q_tile, 1>();

    k_tile    (&k_smem)[1]           = al.allocate<k_tile, 1          >();

    v_tile    (&v_smem)[1]           = al.allocate<v_tile, 1          >();

    l_col_vec (&l_smem)[1] = al.allocate<l_col_vec, 1>();

    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x;

    int32_t* q2k_block_sparse_index_ptr = g.q2k_block_sparse_index + blockIdx.z * gridDim.y * gridDim.x * g.max_kv_blocks_per_q + blockIdx.y * gridDim.x * g.max_kv_blocks_per_q  + blockIdx.x * g.max_kv_blocks_per_q;
    int32_t* q2k_block_sparse_num_ptr = g.q2k_block_sparse_num + blockIdx.z * gridDim.y * gridDim.x  + blockIdx.y * gridDim.x + blockIdx.x;
    int32_t kv_blocks   = q2k_block_sparse_num_ptr[0];
    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived, v_smem_arrived;
    if (threadIdx.x == 0) {
        int32_t kv_block_index = q2k_block_sparse_index_ptr[0];

        init_semaphore(qsmem_semaphore, 0, 1);
        init_semaphore(k_smem_arrived, 0, 1);
        init_semaphore(v_smem_arrived, 0, 1);

        // preload q block
        coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
        tma::load_async(q_smem[0], g.q, q_tile_idx, qsmem_semaphore);

        // preload the zeroth block of kv
        tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
        coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
        tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

        tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
        coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
        tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
    }
    __syncthreads();

    rt_fl<16, 64>  att_block;
    rt_bf<16, 64>  att_block_mma;
    rt_fl<16, K::tile_width> o_reg;

    col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);

    // wait for q block
    wait(qsmem_semaphore, 0);

    for (int kv_idx = 0; kv_idx < kv_blocks - 1; kv_idx++) {
        // preload kv index
        int32_t kv_block_index = q2k_block_sparse_index_ptr[kv_idx + 1];

        // wait k
        wait(k_smem_arrived, kv_idx % 2);

        // compute QK^T
        warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

        copy(max_vec_last_scaled, max_vec);
        if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
        else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }

        warpgroup::mma_async_wait();

        // load K
        if (threadIdx.x == 0) {
            tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
            coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
            tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);
        }

        // exp
        right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]], base_types::constants<float>::neg_infty());
        row_max(max_vec, att_block, max_vec);

        if constexpr (D == 64) {
            mul(att_block, att_block,    1.44269504089f*0.125f);
            mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
        }
        else                   {
            mul(att_block, att_block,    1.44269504089f*0.08838834764f);
            mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
        }

        sub_row(att_block, att_block, max_vec_scaled);
        exp2(att_block, att_block);
        sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
        exp2(max_vec_last_scaled,       max_vec_last_scaled);
        mul(norm_vec,            norm_vec,     max_vec_last_scaled);
        row_sum(norm_vec,  att_block, norm_vec);
        add(att_block, att_block, 0.f);
        copy(att_block_mma, att_block);
        mul_row(o_reg, o_reg, max_vec_last_scaled);

        // wait v
        wait(v_smem_arrived, kv_idx % 2);

        // compute SV
        warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
        warpgroup::mma_async_wait();

        // load V
        if (threadIdx.x == 0) {
            tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
            coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
            tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
        }
    }

    // last iter
    {
        int kv_idx = kv_blocks - 1;
        // wait k
        wait(k_smem_arrived, kv_idx % 2);

        // compute QK^T
        warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

        copy(max_vec_last_scaled, max_vec);
        if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
        else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }

        warpgroup::mma_async_wait();

        // exp
        right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]], base_types::constants<float>::neg_infty());

        row_max(max_vec, att_block, max_vec);

        if constexpr (D == 64) {
            mul(att_block, att_block,    1.44269504089f*0.125f);
            mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
        }
        else                   {
            mul(att_block, att_block,    1.44269504089f*0.08838834764f);
            mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
        }

        sub_row(att_block, att_block, max_vec_scaled);
        exp2(att_block, att_block);
        sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
        exp2(max_vec_last_scaled,       max_vec_last_scaled);
        mul(norm_vec,            norm_vec,     max_vec_last_scaled);
        row_sum(norm_vec,  att_block, norm_vec);
        add(att_block, att_block, 0.f);
        copy(att_block_mma, att_block);
        mul_row(o_reg, o_reg, max_vec_last_scaled);

        // wait v
        wait(v_smem_arrived, kv_idx % 2);

        // compute SV
        warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
        warpgroup::mma_async_wait();
    }

    div_row(o_reg, o_reg, norm_vec);
    warpgroup::store(o_smem[0], o_reg);
    __syncthreads();

    // TK store_async internally calls syncwarp so we need to route on warp level
    if (threadIdx.x / 32 == 0) {
        coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
        tma::store_async(g.o, o_smem[0], o_tile_idx);
    }

    mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec_scaled);

    if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
    else                   { mul(norm_vec, norm_vec, -11.313708499f); }

    warpgroup::store(l_smem[0], norm_vec);
    __syncthreads();

    if (threadIdx.x / 32 == 0) {
        coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, seq_idx};
        tma::store_async(g.l, l_smem[0], tile_idx);
    }
    tma::store_async_wait();
#endif  // FASTVIDEO_TK_HOPPER
}
