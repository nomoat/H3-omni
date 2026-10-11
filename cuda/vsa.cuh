// FastH3 VSA-H3 orchestration: 4x4x4 video tiles, dense text/audio,
// FP32 pooled selection and compression, learned BF16 gate, Hopper attention.
// Included after the core tensor helpers in h3_gpu_cuda.cu.
#include <cub/block/block_radix_sort.cuh>
#include "vsa_hopper.h"

struct h3_gpu_vsa {
    h3_gpu *gpu;
    int rows, heads, tiles, prefix, keep;
    h3_gpu_tensor *map{}, *sizes{}, *q{}, *k{}, *v{}, *out{}, *lse{};
    h3_gpu_tensor *poolq{}, *poolk{}, *poolv{}, *scores{}, *compression{};
    h3_gpu_tensor *indices{}, *counts{};
};

namespace h3_vsa {
constexpr int D = 128, TILE = 64, MAX_TILES = 4096;

__global__ void tile_pool(const ushort *src, ushort *tiled, float *pooled,
                          const int *map, const int *sizes, int tiles, int heads) {
    int tile = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    // Match ATen 2.9.1 sum(dim=2): four reduction lanes, each with four
    // independent accumulators. A sequential sum changes rare BF16 ties.
    float lane_sum[4];
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        float acc[4] = {};
        #pragma unroll
        for (int round = 0; round < 4; ++round) {
            #pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                int i = lane + slot * 4 + round * 16;
                int row = map[tile * TILE + i];
                ushort x = row < 0 ? 0 : src[((size_t)row * heads + head) * D + d];
                tiled[((size_t)head * tiles * TILE + tile * TILE + i) * D + d] = x;
                acc[slot] += h3_bf16_to_f32(x);
            }
        }
        lane_sum[lane] = ((acc[0] + acc[1]) + acc[2]) + acc[3];
    }
    float sum = (lane_sum[0] + lane_sum[2]) + (lane_sum[1] + lane_sum[3]);
    pooled[((size_t)head * tiles + tile) * D + d] = sum / sizes[tile];
}

// Sort scores only for video columns. Indices are emitted in ascending tile
// order, matching a bool-mask compaction and the upstream accumulation order.
__global__ void select_blocks(const float *scores, int *indices, int *counts,
                              int tiles, int prefix, int keep) {
    using Sort = cub::BlockRadixSort<float, 256, 16, int>;
    __shared__ typename Sort::TempStorage sort;
    __shared__ int chosen[MAX_TILES];
    int qtile = blockIdx.x, head = blockIdx.y;
    size_t row = (size_t)head * tiles + qtile;
    if (qtile < prefix) {
        for (int j = threadIdx.x; j < tiles; j += blockDim.x) indices[row * tiles + j] = j;
        if (!threadIdx.x) counts[row] = tiles;
        return;
    }
    float values[16]; int ids[16];
    for (int i = 0; i < 16; ++i) {
        int j = threadIdx.x * 16 + i;
        values[i] = j >= prefix && j < tiles ? scores[row * tiles + j] : -INFINITY;
        ids[i] = j;
        chosen[j] = j < prefix;
    }
    __syncthreads();
    Sort(sort).SortDescending(values, ids);
    for (int i = 0; i < 16; ++i)
        if (threadIdx.x * 16 + i < keep) chosen[ids[i]] = 1;
    __syncthreads();
    if (!threadIdx.x) {
        int count = 0;
        for (int j = 0; j < tiles; ++j) if (chosen[j]) indices[row * tiles + count++] = j;
        counts[row] = count;
    }
}

// ATen's persistent softmax sums each lane's strided elements first,
// then reduces with XOR shuffles. Keep this ordering through 2048 tiles.
template<int CAPACITY>
__global__ void pooled_softmax_warp(float *scores, int tiles, int rows) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    float values[CAPACITY / 32], largest = -INFINITY;
    #pragma unroll
    for (int i = 0; i < CAPACITY / 32; ++i) {
        int col = lane + i * 32;
        values[i] = col < tiles ? scores[(size_t)row * tiles + col] : -INFINITY;
        largest = fmaxf(largest, values[i]);
    }
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, offset));
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < CAPACITY / 32; ++i) {
        values[i] = expf(values[i] - largest);
        sum += values[i];
    }
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    #pragma unroll
    for (int i = 0; i < CAPACITY / 32; ++i) {
        int col = lane + i * 32;
        if (col < tiles) scores[(size_t)row * tiles + col] = values[i] / sum;
    }
}

template<bool MAXIMUM>
__device__ float reduce_softmax_block(float value, float *shared) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    #pragma unroll
    for (int offset = 16; offset; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = MAXIMUM ? fmaxf(value, other) : value + other;
    }
    __syncthreads();
    if (!lane) shared[warp] = value;
    __syncthreads();
    if (!warp) {
        value = shared[lane];
        #pragma unroll
        for (int offset = 16; offset; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, value, offset);
            value = MAXIMUM ? fmaxf(value, other) : value + other;
        }
        if (!lane) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

// Above 2048, ATen uses 1024 threads and ceil(tiles/1024) registers per
// thread, with a two-level shuffle-down reduction (maximum: 4096 tiles).
__global__ void pooled_softmax_block(float *scores, int tiles) {
    __shared__ float shared[32];
    float values[4], largest = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int col = threadIdx.x + i * 1024;
        values[i] = col < tiles ? scores[(size_t)blockIdx.x * tiles + col] : -INFINITY;
        largest = fmaxf(largest, values[i]);
    }
    largest = reduce_softmax_block<true>(largest, shared);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (threadIdx.x + i * 1024 < tiles) sum += expf(values[i] - largest);
    sum = reduce_softmax_block<false>(sum, shared);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int col = threadIdx.x + i * 1024;
        if (col < tiles) scores[(size_t)blockIdx.x * tiles + col] = expf(values[i] - largest) / sum;
    }
}

static void pooled_softmax(float *scores, int tiles, int heads, cudaStream_t stream) {
    int rows = tiles * heads;
    #define WARP_CASE(N) if (tiles <= N) { \
        pooled_softmax_warp<N><<<(rows + 3) / 4, 128, 0, stream>>>(scores, tiles, rows); return; }
    WARP_CASE(32) WARP_CASE(64) WARP_CASE(128) WARP_CASE(256)
    WARP_CASE(512) WARP_CASE(1024) WARP_CASE(2048)
    #undef WARP_CASE
    pooled_softmax_block<<<rows, 1024, 0, stream>>>(scores, tiles);
}

__global__ void untile_merge(ushort *output, const ushort *tiled, const ushort *gate,
                             const float *compression, const int *map, int tiles, int heads) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)tiles * TILE * heads * D;
    if (i >= total) return;
    int d = i % D, head = (i / D) % heads, padded_row = i / (D * heads);
    int row = map[padded_row];
    if (row < 0) return;
    float x = h3_bf16_to_f32(tiled[((size_t)head * tiles * TILE + padded_row) * D + d]);
    size_t dst = ((size_t)row * heads + head) * D + d;
    // FastVideo casts pooled output to BF16 before the fused addcmul.
    float c = bf16_round(compression[((size_t)head * tiles + padded_row / TILE) * D + d]);
    output[dst] = h3_f32_to_bf16(fmaf(c, h3_bf16_to_f32(gate[dst]), x));
}
} // namespace h3_vsa

extern "C" void h3_gpu_vsa_free(h3_gpu_vsa *v) {
    if (!v) return;
    for (auto *t : {v->map, v->sizes, v->q, v->k, v->v, v->out, v->lse,
                   v->poolq, v->poolk, v->poolv, v->scores, v->compression, v->indices, v->counts})
        h3_gpu_tensor_free(t);
    delete v;
}

extern "C" h3_gpu_vsa *h3_gpu_vsa_new(h3_gpu *g, unsigned text, unsigned audio,
                                       unsigned time, unsigned height, unsigned width,
                                       unsigned heads) {
    if (!g || !text || !audio || !time || !height || !width || !heads) return nullptr;
    // Bound dimensions before constructing maps or multiplying allocation sizes.
    const uint64_t blocks = ((uint64_t)text + 63) / 64 + ((uint64_t)audio + 63) / 64;
    const uint64_t nt = ((uint64_t)time + 3) / 4, nh = ((uint64_t)height + 3) / 4;
    const uint64_t nw = ((uint64_t)width + 3) / 4;
    if (blocks >= h3_vsa::MAX_TILES || nt > h3_vsa::MAX_TILES ||
        nh > h3_vsa::MAX_TILES || nw > h3_vsa::MAX_TILES ||
        blocks + nt * nh * nw > h3_vsa::MAX_TILES || heads > 56) {
        h3_gpu_set_error(g, "VSA requires at most %d tiles and 56 heads", h3_vsa::MAX_TILES);
        return nullptr;
    }
    std::vector<int> map, sizes;
    try {
        map.reserve((blocks + nt * nh * nw) * 64);
        sizes.reserve(blocks + nt * nh * nw);
    } catch (const std::bad_alloc &) {
        h3_gpu_set_error(g, "out of host memory constructing VSA tiles");
        return nullptr;
    }
    int cursor = 0, prefix = 0;
    for (unsigned count : {text, audio}) {
        for (unsigned i = 0; i < count; i += 64) {
            int valid = std::min(64U, count - i);
            sizes.push_back(valid); ++prefix;
            for (int j = 0; j < 64; ++j) map.push_back(j < valid ? cursor + (int)i + j : -1);
        }
        cursor += count;
    }
    for (unsigned t = 0; t < time; t += 4)
        for (unsigned y = 0; y < height; y += 4)
            for (unsigned x = 0; x < width; x += 4) {
                int valid = 0;
                for (unsigned dt = t; dt < std::min(time, t + 4); ++dt)
                    for (unsigned dy = y; dy < std::min(height, y + 4); ++dy)
                        for (unsigned dx = x; dx < std::min(width, x + 4); ++dx) {
                            map.push_back(cursor + (dt * height + dy) * width + dx); ++valid;
                        }
                sizes.push_back(valid);
                for (int j = valid; j < 64; ++j) map.push_back(-1);
            }
    if (sizes.size() > h3_vsa::MAX_TILES) {
        h3_gpu_set_error(g, "VSA supports at most %d tiles", h3_vsa::MAX_TILES); return nullptr;
    }
    auto *v = new (std::nothrow) h3_gpu_vsa{};
    if (!v) return nullptr;
    v->gpu = g; v->rows = cursor + time * height * width; v->heads = heads;
    v->tiles = sizes.size(); v->prefix = prefix;
    v->keep = std::max(1, (v->tiles - prefix + 9) / 10);
    size_t n = map.size() * heads * 128, pool = sizes.size() * heads * 128;
    v->map = allocate(g, map.size(), H3_GPU_F32); v->sizes = allocate(g, sizes.size(), H3_GPU_F32);
    v->q = allocate(g, n, H3_GPU_BF16); v->k = allocate(g, n, H3_GPU_BF16);
    v->v = allocate(g, n, H3_GPU_BF16); v->out = allocate(g, n, H3_GPU_BF16);
    v->lse = allocate(g, map.size() * heads, H3_GPU_F32);
    v->poolq = allocate(g, pool, H3_GPU_F32); v->poolk = allocate(g, pool, H3_GPU_F32);
    v->poolv = allocate(g, pool, H3_GPU_F32); v->compression = allocate(g, pool, H3_GPU_F32);
    v->scores = allocate(g, sizes.size() * sizes.size() * heads, H3_GPU_F32);
    v->indices = allocate(g, sizes.size() * sizes.size() * heads, H3_GPU_F32);
    v->counts = allocate(g, sizes.size() * heads, H3_GPU_F32);
    if (!v->map || !v->sizes || !v->q || !v->k || !v->v || !v->out || !v->lse ||
        !v->poolq || !v->poolk || !v->poolv || !v->compression || !v->scores || !v->indices || !v->counts ||
        !cuda_ok(g, cudaMemcpyAsync(v->map->data, map.data(), map.size() * sizeof(int), cudaMemcpyHostToDevice, g->stream)) ||
        !cuda_ok(g, cudaMemcpyAsync(v->sizes->data, sizes.data(), sizes.size() * sizeof(int), cudaMemcpyHostToDevice, g->stream)) ||
        !cuda_ok(g, cudaStreamSynchronize(g->stream))) {
        h3_gpu_vsa_free(v); return nullptr;
    }
    return v;
}

extern "C" int h3_gpu_vsa_bf16(h3_gpu_vsa *v, h3_gpu_tensor *out,
                                 const h3_gpu_tensor *q, const h3_gpu_tensor *k,
                                 const h3_gpu_tensor *value, const h3_gpu_tensor *gate) {
    if (!v) return 0;
    h3_gpu *g = v->gpu;
    size_t elements = (size_t)v->rows * v->heads * 128;
    const h3_gpu_tensor *tensors[] = {out, q, k, value, gate};
    for (auto *t : tensors)
        if (!h3_gpu_require_bf16(g, t, elements, "VSA input/output")) return 0;
    dim3 grid(v->tiles, v->heads);
    const h3_gpu_tensor *src[] = {q, k, value};
    h3_gpu_tensor *dst[] = {v->q, v->k, v->v}, *pool[] = {v->poolq, v->poolk, v->poolv};
    for (int i = 0; i < 3; ++i)
        h3_vsa::tile_pool<<<grid, 128, 0, g->stream>>>((const ushort *)src[i]->data,
            (ushort *)dst[i]->data, (float *)pool[i]->data, (const int *)v->map->data,
            (const int *)v->sizes->data, v->tiles, v->heads);
    float scale = 1.0f / sqrtf(128.0f), one = 1.0f, zero = 0.0f;
    int n = v->tiles; long long stride = (long long)n * 128;
    if (!launch_ok(g) || !blas_ok(g, cublasSgemmStridedBatched(g->blas, CUBLAS_OP_T, CUBLAS_OP_N,
            n, n, 128, &scale, (const float *)v->poolk->data, 128, stride,
            (const float *)v->poolq->data, 128, stride, &zero,
            (float *)v->scores->data, n, (long long)n * n, v->heads))) return 0;
    h3_vsa::select_blocks<<<grid, 256, 0, g->stream>>>((const float *)v->scores->data,
        (int *)v->indices->data, (int *)v->counts->data, n, v->prefix, v->keep);
    if (!launch_ok(g) || !cuda_ok(g, h3_vsa_hopper(v->out->data, v->q->data, v->k->data, v->v->data,
            (float *)v->lse->data, (int *)v->indices->data, (int *)v->counts->data,
            (int *)v->sizes->data, n, v->heads, g->stream))) return 0;
    h3_vsa::pooled_softmax((float *)v->scores->data, n, v->heads, g->stream);
    if (!launch_ok(g) || !blas_ok(g, cublasSgemmStridedBatched(g->blas, CUBLAS_OP_N, CUBLAS_OP_N,
            128, n, n, &one, (const float *)v->poolv->data, 128, stride,
            (const float *)v->scores->data, n, (long long)n * n, &zero,
            (float *)v->compression->data, 128, stride, v->heads))) return 0;
    size_t total = (size_t)n * 64 * v->heads * 128;
    h3_vsa::untile_merge<<<(total + 255) / 256, 256, 0, g->stream>>>((ushort *)out->data,
        (const ushort *)v->out->data, (const ushort *)gate->data, (const float *)v->compression->data,
        (const int *)v->map->data, n, v->heads);
    return launch_ok(g);
}
