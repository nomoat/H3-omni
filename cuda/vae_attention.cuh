// Fused non-causal attention for the video VAE (head dim 64, FP32 in and out)
// on 3xTF32 tensor-core products. Q, K, V and the output use the token-major
// layout [token][head][64], so no head packing is needed.
//
// Two kernels:
// 1. split_kv: K and V are split once into TF32 high/low parts and stored in
//    the order of the m16n8k8 B fragments (per head, padded to 64 keys).
// 2. attention: one CTA = 16 * WARPS query rows of one head, 16 rows per
//    warp. Key blocks of 64 are copied to shared memory with cp.async,
//    double-buffered. Both products (Q K^T and P V) use
//    x*y ~= xl*yh + xh*yl + xh*yh (3xTF32) with an online FP32 softmax (expf).
//    Each key block's P V goes into a fresh accumulator that is added to O
//    with FP32 adds (PROMOTE): the Hopper tensor-core accumulator is not a full
//    IEEE accumulator, and carrying the running sum over all keys in it costs
//    10x accuracy when attention is nearly uniform.
// TERMS=1 (plain TF32), TERMS=4 (adds xl*yl), PROMOTE=false and
// INTERLEAVE=false (same values, other instruction order) exist for the study only.
#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace vae_attention {
constexpr int D = 64, BN = 64;
// Fragment-ordered high/low data, one uint4 per entry:
//   K: [head][key][kk][t]   -> {h(8kk+t), h(8kk+t+4), l(8kk+t), l(8kk+t+4)}
//   V: [head][pair][dim]    -> {h(key 2p), h(key 2p+1), l(key 2p), l(key 2p+1)}
// Both are 128 words per key. In shared memory the rows are padded to 144
// (K) and 264 (V) words so that 128-bit fragment loads are conflict-free.
constexpr int KSTRIDE = 144, VSTRIDE = 264;
constexpr size_t STAGE_WORDS = size_t(BN) * KSTRIDE + size_t(BN / 2) * VSTRIDE;
constexpr size_t SMEM = 2 * STAGE_WORDS * sizeof(unsigned);

__host__ __device__ inline int padded_keys(int n) { return (n + BN - 1) / BN * BN; }
// Words of each of the K and V split buffers.
inline size_t split_words(int n, int heads) { return size_t(heads) * padded_keys(n) * 2 * D; }

__device__ __forceinline__ unsigned to_tf32(float x) {
    unsigned bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x));
    return bits;
}
__device__ __forceinline__ void split(float x, unsigned &high, unsigned &low) {
    high = to_tf32(x);
    low = to_tf32(x - __uint_as_float(high));
}
__device__ __forceinline__ void mma(float c[4], const unsigned a[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// c += a * b with b = {b0 high, b1 high, b0 low, b1 low}.
template <int TERMS>
__device__ __forceinline__ void mma3(float c[4], const unsigned ah[4], const unsigned al[4],
                                     uint4 b) {
    if (TERMS >= 3) {
        if (TERMS == 4) mma(c, al, b.z, b.w);
        mma(c, al, b.x, b.y);
        mma(c, ah, b.z, b.w);
    }
    mma(c, ah, b.x, b.y);
}
__device__ __forceinline__ void copy16(void *shared, const void *global) {
    unsigned at = unsigned(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(at), "l"(global));
}

// One thread per uint4 entry of K and of V; keys past n are zero.
__global__ void split_kv(uint4 *__restrict__ ks, uint4 *__restrict__ vs,
                         const float *__restrict__ k, const float *__restrict__ v, int n,
                         int heads) {
    const int keys = padded_keys(n);
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(heads) * keys * 32) return;
    const size_t row_stride = size_t(heads) * D;
    // K entry (head, key, kk, t): consecutive threads read consecutive dims.
    {
        int t = i % 4, kk = i / 4 % 8, key = int(i / 32 % keys), head = int(i / 32 / keys);
        float x0 = 0.f, x1 = 0.f;
        if (key < n) {
            const float *src = k + key * row_stride + head * D + 8 * kk + t;
            x0 = src[0];
            x1 = src[4];
        }
        uint4 w;
        split(x0, w.x, w.z);
        split(x1, w.y, w.w);
        ks[i] = w;
    }
    // V entry (head, pair, dim): the same number of entries as K.
    {
        int dim = i % D, pair = int(i / D % (keys / 2)), head = int(i / D / (keys / 2));
        float x0 = 0.f, x1 = 0.f;
        const float *src = v + size_t(2 * pair) * row_stride + head * D + dim;
        if (2 * pair < n) x0 = src[0];
        if (2 * pair + 1 < n) x1 = src[row_stride];
        uint4 w;
        split(x0, w.x, w.z);
        split(x1, w.y, w.w);
        vs[i] = w;
    }
}

// m16n8k8 TF32 fragments (g = lane / 4, t = lane % 4):
//   A: a0 (g, t), a1 (g+8, t), a2 (g, t+4), a3 (g+8, t+4)
//   B: b0 (k=t, n=g), b1 (k=t+4, n=g)
//   C: c0 (g, 2t), c1 (g, 2t+1), c2 (g+8, 2t), c3 (g+8, 2t+1)
// For P V the key order inside each group of 8 is permuted so that the C
// fragment of S is directly the A fragment of P: k index t <-> key 2t and
// k index t+4 <-> key 2t+1; V is read in the same order.
template <int TERMS, bool PROMOTE, int WARPS, bool INTERLEAVE>
__global__ void __launch_bounds__(WARPS * 32)
attention(float *__restrict__ y, const float *__restrict__ q, const uint4 *__restrict__ ksplit,
          const uint4 *__restrict__ vsplit, int n, int heads, float scale) {
    constexpr int THREADS = WARPS * 32;
    extern __shared__ uint4 smem4[];
    unsigned *stage = reinterpret_cast<unsigned *>(smem4);
    const int head = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane / 4, t = lane % 4, keys = padded_keys(n);
    const size_t row_stride = size_t(heads) * D;
    const int row0 = blockIdx.x * (16 * WARPS) + warp * 16 + g, row1 = row0 + 8;
    const uint4 *kh = ksplit + size_t(head) * keys * 32, *vh = vsplit + size_t(head) * keys * 32;
    // Block b: 64 K rows of 32 entries and 32 V pair rows of 64 entries.
    auto load = [&](int block, int buffer) {
        unsigned *ks = stage + buffer * STAGE_WORDS, *vs = ks + BN * KSTRIDE;
        const uint4 *kg = kh + size_t(block) * BN * 32, *vg = vh + size_t(block) * BN * 32;
#pragma unroll
        for (int r = 0; r < BN * 32 / THREADS; r++) {
            int e = threadIdx.x + r * THREADS;
            copy16(ks + (e / 32) * KSTRIDE + (e % 32) * 4, kg + e);
            copy16(vs + (e / 64) * VSTRIDE + (e % 64) * 4, vg + e);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    load(0, 0);

    unsigned qh[8][4], ql[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        float a[4];
        a[0] = row0 < n ? q[row0 * row_stride + head * D + 8 * kk + t] : 0.f;
        a[1] = row1 < n ? q[row1 * row_stride + head * D + 8 * kk + t] : 0.f;
        a[2] = row0 < n ? q[row0 * row_stride + head * D + 8 * kk + t + 4] : 0.f;
        a[3] = row1 < n ? q[row1 * row_stride + head * D + 8 * kk + t + 4] : 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) split(a[i], qh[kk][i], ql[kk][i]);
    }
    float o[8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    const int blocks = keys / BN;
    for (int block = 0; block < blocks; block++) {
        const int key0 = block * BN;
        if (block + 1 < blocks) {
            load(block + 1, (block + 1) % 2);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned *ks = stage + (block % 2) * STAGE_WORDS, *vs = ks + BN * KSTRIDE;

        // S = Q K^T. INTERLEAVE runs the 8 independent accumulators side by
        // side (kk outer); each accumulator still sums over kk in the same order.
        float s[8][4] = {};
        if (INTERLEAVE) {
#pragma unroll
            for (int kk = 0; kk < 8; kk++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint4 b = *reinterpret_cast<const uint4 *>(ks + (8 * j + g) * KSTRIDE +
                                                               kk * 16 + 4 * t);
                    mma3<TERMS>(s[j], qh[kk], ql[kk], b);
                }
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    uint4 b = *reinterpret_cast<const uint4 *>(ks + (8 * j + g) * KSTRIDE +
                                                               kk * 16 + 4 * t);
                    mma3<TERMS>(s[j], qh[kk], ql[kk], b);
                }
        }
        // Scale, mask the keys past the end, online softmax per row.
        float top[2] = {m[0], m[1]};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int key = key0 + 8 * j + 2 * t + (i & 1);
                s[j][i] = key < n ? s[j][i] * scale : -INFINITY;
                top[i >> 1] = fmaxf(top[i >> 1], s[j][i]);
            }
        float correction[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffffu, top[r], 1));
            top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffffu, top[r], 2));
            correction[r] = expf(m[r] - top[r]);
            m[r] = top[r];
            l[r] *= correction[r];
        }
        // O += P V. INTERLEAVE: key groups j outer, 8 accumulators side by side;
        // each accumulator still sums over j in the same order as below.
        if (INTERLEAVE) {
            float acc[8][4];
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int i = 0; i < 4; i++)
                    acc[nn][i] = PROMOTE ? 0.f : o[nn][i] * correction[i >> 1];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned ph[4], pl[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    s[j][i] = expf(s[j][i] - m[i >> 1]);
                    l[i >> 1] += s[j][i];
                }
                // C fragment (c0 c1 c2 c3) -> A fragment (a0=c0, a1=c2, a2=c1, a3=c3).
                split(s[j][0], ph[0], pl[0]);
                split(s[j][2], ph[1], pl[1]);
                split(s[j][1], ph[2], pl[2]);
                split(s[j][3], ph[3], pl[3]);
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    uint4 b = *reinterpret_cast<const uint4 *>(vs + (4 * j + t) * VSTRIDE +
                                                               4 * (8 * nn + g));
                    mma3<TERMS>(acc[nn], ph, pl, b);
                }
            }
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int i = 0; i < 4; i++)
                    o[nn][i] = PROMOTE ? o[nn][i] * correction[i >> 1] + acc[nn][i] : acc[nn][i];
        } else {
            unsigned ph[8][4], pl[8][4];
#pragma unroll
            for (int j = 0; j < 8; j++) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    s[j][i] = expf(s[j][i] - m[i >> 1]);
                    l[i >> 1] += s[j][i];
                }
                split(s[j][0], ph[j][0], pl[j][0]);
                split(s[j][2], ph[j][1], pl[j][1]);
                split(s[j][1], ph[j][2], pl[j][2]);
                split(s[j][3], ph[j][3], pl[j][3]);
            }
#pragma unroll
            for (int nn = 0; nn < 8; nn++) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                if (!PROMOTE)
#pragma unroll
                    for (int i = 0; i < 4; i++) acc[i] = o[nn][i] * correction[i >> 1];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint4 b = *reinterpret_cast<const uint4 *>(vs + (4 * j + t) * VSTRIDE +
                                                               4 * (8 * nn + g));
                    mma3<TERMS>(acc, ph[j], pl[j], b);
                }
#pragma unroll
                for (int i = 0; i < 4; i++)
                    o[nn][i] = PROMOTE ? o[nn][i] * correction[i >> 1] + acc[i] : acc[i];
            }
        }
        // The next iteration's load overwrites this buffer.
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
#pragma unroll
    for (int nn = 0; nn < 8; nn++) {
        int col = head * D + 8 * nn + 2 * t;
        if (row0 < n)
            *reinterpret_cast<float2 *>(y + row0 * row_stride + col) =
                make_float2(o[nn][0] / l[0], o[nn][1] / l[0]);
        if (row1 < n)
            *reinterpret_cast<float2 *>(y + row1 * row_stride + col) =
                make_float2(o[nn][2] / l[1], o[nn][3] / l[1]);
    }
}

// split_k and split_v hold split_words(n, heads) words each (16-byte aligned).
// y needs 8-byte alignment and must not overlap the inputs. Returns the launch error.
template <int TERMS = 3, bool PROMOTE = true, int WARPS = 8, bool INTERLEAVE = true>
inline cudaError_t launch(float *y, const float *q, const float *k, const float *v,
                          unsigned *split_k, unsigned *split_v, int n, int heads, float scale,
                          cudaStream_t stream) {
    static cudaError_t configured = cudaFuncSetAttribute(
        attention<TERMS, PROMOTE, WARPS, INTERLEAVE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(SMEM));
    if (configured != cudaSuccess) return configured;
    size_t entries = size_t(heads) * padded_keys(n) * 32;
    split_kv<<<unsigned((entries + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<uint4 *>(split_k), reinterpret_cast<uint4 *>(split_v), k, v, n, heads);
    dim3 grid((n + 16 * WARPS - 1) / (16 * WARPS), heads);
    attention<TERMS, PROMOTE, WARPS, INTERLEAVE><<<grid, WARPS * 32, SMEM, stream>>>(
        y, q, reinterpret_cast<const uint4 *>(split_k), reinterpret_cast<const uint4 *>(split_v),
        n, heads, scale);
    return cudaGetLastError();
}
}  // namespace vae_attention
