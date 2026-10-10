// Bit-exactness check and timing of the hand-written DiT elementwise kernels
// (cuda/fast_primitives.cuh) against the generated ones (cuda/primitives.cuh)
// with the generated launch shapes. Random BF16 inputs; DiT shapes at a 10 s
// request (29700 rows) plus small odd shapes for bounds and tails.
// Build: nvcc -O2 -std=c++17 -arch=sm_90 --fmad=false -I. -Icuda
//        tests/serving/profile/elementwise_bench.cu -o elementwise_bench
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
extern "C" {
#include "h3_gpu.h"
#include "h3_metal.h"
}
using uint = unsigned int;
using ushort = unsigned short;
#include "primitives.cuh"
#include "fast_primitives.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void fill_bf16(ushort *p, size_t n, uint seed, float scale, float offset) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint h = (uint)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        uint h2 = h * 668265263u ^ (h >> 7);
        // Sum of two uniforms: a rough bell shape; a few large values.
        float u = ((h & 0xffffff) / 16777216.0f) + ((h2 & 0xffffff) / 16777216.0f) - 1.0f;
        if ((h2 >> 24) == 0) u *= 40.0f;
        p[i] = h3_f32_to_bf16(u * scale + offset);
    }
}
static ushort *alloc_fill(size_t n, uint seed, float scale, float offset = 0) {
    ushort *p;
    CK(cudaMalloc(&p, n * 2 + 16));
    fill_bf16<<<2048, 256>>>(p, n, seed, scale, offset);
    CK(cudaGetLastError());
    return p;
}
static uint *row_map(uint rows, uint mod_rows) {
    std::vector<uint> h(rows);
    for (uint r = 0; r < rows; ++r) h[r] = (uint)((uint64_t)r * mod_rows / rows);
    uint *d;
    CK(cudaMalloc(&d, rows * 4));
    CK(cudaMemcpy(d, h.data(), rows * 4, cudaMemcpyHostToDevice));
    return d;
}
static int failures = 0;
static void compare(const char *what, const ushort *a, const ushort *b, size_t n) {
    std::vector<ushort> x(n), y(n);
    CK(cudaMemcpy(x.data(), a, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n * 2, cudaMemcpyDeviceToHost));
    size_t diff = 0, first = n;
    for (size_t i = 0; i < n; ++i)
        if (x[i] != y[i]) { if (!diff) first = i; ++diff; }
    printf("  %-28s %s (%zu of %zu differ", what, diff ? "MISMATCH" : "identical", diff, n);
    if (diff) printf(", first %zu: %04x vs %04x", first, x[first], y[first]);
    printf(")\n");
    failures += diff != 0;
}
template <class F> static float time_ms(F f, int reps = 20) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    CK(cudaGetLastError());
    return ms / reps;
}
static void report(const char *what, float gen, float fast, double bytes) {
    printf("  %-28s generated %.3f ms, fast %.3f ms (%.2fx), fast %.0f GB/s\n", what, gen, fast,
           gen / fast, bytes / fast / 1e6);
}

static void dit_case(uint rows, uint width, uint ffn, uint heads, uint mod_rows, uint half,
                     bool timing) {
    const uint slots = 6, dim = 128;
    printf("rows %u width %u ffn %u heads %u rope_half %u\n", rows, width, ffn, heads, half);
    size_t n = (size_t)rows * width;
    uint *map = row_map(rows, mod_rows);
    ushort *mod = alloc_fill((size_t)mod_rows * slots * width, 11, 0.5f);
    ushort *mod2 = alloc_fill((size_t)mod_rows * slots * width, 12, 0.5f);
    ushort *weight = alloc_fill(width, 13, 0.3f, 1.0f);
    ushort *hidden0 = alloc_fill(n, 14, 3.0f);
    ushort *branch = alloc_fill(n, 15, 2.0f);
    ushort *res_a, *res_b, *y_a, *y_b;
    CK(cudaMalloc(&res_a, n * 2)); CK(cudaMalloc(&res_b, n * 2));
    CK(cudaMalloc(&y_a, n * 2)); CK(cudaMalloc(&y_b, n * 2));

    // AdaLN alone (slots 0, 1).
    adaln_args aa{rows, width, slots, 0, 1, 1e-5f};
    fast_gate_adaln_args fa{rows, width, slots, 0, 0, 1, 1e-5f};
    auto gen_adaln = [&] { h3_adaln_bf16<<<rows, 256>>>(hidden0, weight, mod, map, y_a, aa); };
    auto fast_adaln = [&] {
        h3_fast_gate_adaln_bf16<11, false><<<rows, 128>>>(hidden0, nullptr, nullptr, nullptr,
                                                          weight, mod, map, y_b, fa);
    };
    gen_adaln(); fast_adaln(); CK(cudaDeviceSynchronize());
    compare("adaln", y_a, y_b, n);
    if (timing) report("adaln", time_ms(gen_adaln), time_ms(fast_adaln), 4.0 * n);

    // Gate alone, in place as in the DiT (output == residual).
    gate_args ga{rows, width, slots, 2};
    dim3 g2((width + 31) / 32, (rows + 7) / 8), b2(32, 8);
    CK(cudaMemcpy(res_a, hidden0, n * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(res_b, hidden0, n * 2, cudaMemcpyDeviceToDevice));
    h3_gate_bf16<<<g2, b2>>>(res_a, branch, mod, map, res_a, ga);
    h3_fast_gate_bf16<<<(unsigned)((n / 8 + 255) / 256), 256>>>(res_b, branch, mod, map, res_b, ga);
    CK(cudaDeviceSynchronize());
    compare("gate (in place)", res_a, res_b, n);
    if (timing)
        report("gate", time_ms([&] { h3_gate_bf16<<<g2, b2>>>(hidden0, branch, mod, map, res_a, ga); }),
               time_ms([&] {
                   h3_fast_gate_bf16<<<(unsigned)((n / 8 + 255) / 256), 256>>>(hidden0, branch, mod,
                                                                               map, res_b, ga);
               }), 6.0 * n);

    // Gate + AdaLN: generated two kernels vs fused, in place, separate modulations.
    adaln_args aa2{rows, width, slots, 3, 4, 1e-5f};
    fast_gate_adaln_args fga{rows, width, slots, 2, 3, 4, 1e-5f};
    CK(cudaMemcpy(res_a, hidden0, n * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(res_b, hidden0, n * 2, cudaMemcpyDeviceToDevice));
    h3_gate_bf16<<<g2, b2>>>(res_a, branch, mod, map, res_a, ga);
    h3_adaln_bf16<<<rows, 256>>>(res_a, weight, mod2, map, y_a, aa2);
    h3_fast_gate_adaln_bf16<11, true><<<rows, 128>>>(res_b, branch, mod, res_b, weight, mod2, map,
                                                     y_b, fga);
    CK(cudaDeviceSynchronize());
    compare("gate_adaln residual", res_a, res_b, n);
    compare("gate_adaln output", y_a, y_b, n);
    // Other template sizes give the same result.
    h3_fast_gate_adaln_bf16<16, false><<<rows, 128>>>(hidden0, nullptr, nullptr, nullptr, weight,
                                                      mod, map, y_b, fa);
    gen_adaln(); CK(cudaDeviceSynchronize());
    compare("adaln (CHUNKS 16)", y_a, y_b, n);
    if (timing)
        report("gate+adaln", time_ms([&] {
            h3_gate_bf16<<<g2, b2>>>(hidden0, branch, mod, map, res_a, ga);
            h3_adaln_bf16<<<rows, 256>>>(res_a, weight, mod2, map, y_a, aa2);
        }), time_ms([&] {
            h3_fast_gate_adaln_bf16<11, true><<<rows, 128>>>(hidden0, branch, mod, res_b, weight,
                                                             mod2, map, y_b, fga);
        }), 8.0 * n);

    // SwiGLU.
    size_t nf = (size_t)rows * ffn;
    ushort *fc1 = alloc_fill(nf * 2, 16, 3.0f);
    ushort *act_a, *act_b;
    CK(cudaMalloc(&act_a, nf * 2)); CK(cudaMalloc(&act_b, nf * 2));
    swiglu_args sa{rows, ffn};
    dim3 gs((ffn + 31) / 32, (rows + 7) / 8);
    auto gen_sw = [&] { h3_swiglu_bf16<<<gs, b2>>>(fc1, act_a, sa); };
    auto fast_sw = [&] { h3_fast_swiglu_bf16<<<(unsigned)((nf / 8 + 255) / 256), 256>>>(fc1, act_b, sa); };
    gen_sw(); fast_sw(); CK(cudaDeviceSynchronize());
    compare("swiglu", act_a, act_b, nf);
    if (timing) report("swiglu", time_ms(gen_sw), time_ms(fast_sw), 6.0 * nf);
    CK(cudaFree(fc1)); CK(cudaFree(act_a)); CK(cudaFree(act_b));

    // QKV norm + RoPE, grouped (DiT) and plain layouts.
    size_t inner = (size_t)heads * dim, nq = (size_t)rows * inner;
    ushort *qkv = alloc_fill(nq * 3, 17, 4.0f);
    ushort *qw = alloc_fill(dim, 18, 0.3f, 1.0f), *kw = alloc_fill(dim, 19, 0.3f, 1.0f);
    ushort *co = alloc_fill((size_t)rows * half, 20, 0.5f), *si = alloc_fill((size_t)rows * half, 21, 0.5f);
    ushort *o[6];
    for (auto &p : o) CK(cudaMalloc(&p, nq * 2));
    for (uint grouped = 0; grouped < 2; ++grouped) {
        qkv_args qa{rows, heads, dim, half, grouped, 1e-5f};
        dim3 gq(dim / 32, (heads + 3) / 4, rows), bq(32, 4, 1);
        size_t items = (size_t)rows * heads;
        auto gen_q = [&] { h3_qkv_rope_bf16<<<gq, bq>>>(qkv, qw, kw, co, si, o[0], o[1], o[2], qa); };
        auto fast_q = [&] {
            h3_fast_qkv_rope_bf16<<<(unsigned)((items + 7) / 8), 256>>>(qkv, qw, kw, co, si, o[3],
                                                                        o[4], o[5], qa);
        };
        gen_q(); fast_q(); CK(cudaDeviceSynchronize());
        compare(grouped ? "qkv_rope grouped q" : "qkv_rope plain q", o[0], o[3], nq);
        compare(grouped ? "qkv_rope grouped k" : "qkv_rope plain k", o[1], o[4], nq);
        compare(grouped ? "qkv_rope grouped v" : "qkv_rope plain v", o[2], o[5], nq);
        if (timing && grouped) report("qkv_rope grouped", time_ms(gen_q), time_ms(fast_q), 4.0 * nq * 3);
    }
    for (auto &p : o) CK(cudaFree(p));
    CK(cudaFree(qkv)); CK(cudaFree(qw)); CK(cudaFree(kw)); CK(cudaFree(co)); CK(cudaFree(si));
    CK(cudaFree(map)); CK(cudaFree(mod)); CK(cudaFree(mod2)); CK(cudaFree(weight));
    CK(cudaFree(hidden0)); CK(cudaFree(branch));
    CK(cudaFree(res_a)); CK(cudaFree(res_b)); CK(cudaFree(y_a)); CK(cudaFree(y_b));
}

int main() {
    dit_case(29700, 5376, 14336, 56, 61, 48, true);   // 10 s request
    dit_case(7, 5376, 14336, 56, 3, 48, false);       // tiny, partial blocks
    dit_case(1001, 3072, 8192, 24, 5, 32, false);     // other widths, 6 chunks
    dit_case(333, 5376 + 8, 1032, 5, 2, 64, false);   // width tail inside a chunk
    printf("%s\n", failures ? "FAILED" : "ALL IDENTICAL");
    return failures != 0;
}
