// FlashAttention-2 (current adapter) versus FlashAttention-3 on the DiT
// joint-attention shape: speed, and error against an FP32 reference on
// sampled query rows. Usage: attn_bench SEQUENCE [HEADS]
#include "h3_flash.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void fill(__nv_bfloat16 *x, size_t n, unsigned seed, float amplitude) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    unsigned h = unsigned(i) * 2654435761u ^ seed * 2246822519u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    x[i] = __float2bfloat16((float(h) / 4294967296.0f * 2.0f - 1.0f) * amplitude);
}

// One block per (sampled row, head): exact softmax(q k^T * scale) v in FP32.
__global__ void reference(const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                          const int *rows, float *out, int sequence, int heads, float scale) {
    extern __shared__ float scores[];
    int sample = blockIdx.x, head = blockIdx.y, row = rows[sample];
    const __nv_bfloat16 *qr = q + (size_t(row) * heads + head) * 128;
    __shared__ float qs[128];
    if (threadIdx.x < 128) qs[threadIdx.x] = __bfloat162float(qr[threadIdx.x]);
    __syncthreads();
    for (int j = threadIdx.x; j < sequence; j += blockDim.x) {
        const __nv_bfloat16 *kr = k + (size_t(j) * heads + head) * 128;
        float s = 0;
        for (int d = 0; d < 128; d++) s += qs[d] * __bfloat162float(kr[d]);
        scores[j] = s * scale;
    }
    __syncthreads();
    __shared__ float maximum, total;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int j = 0; j < sequence; j++) m = fmaxf(m, scores[j]);
        double t = 0;
        for (int j = 0; j < sequence; j++) t += exp(double(scores[j] - m));
        maximum = m; total = float(t);
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        double acc = 0;
        for (int j = 0; j < sequence; j++)
            acc += exp(double(scores[j] - maximum)) *
                   __bfloat162float(v[(size_t(j) * heads + head) * 128 + threadIdx.x]);
        out[(size_t(sample) * heads + head) * 128 + threadIdx.x] = float(acc / total);
    }
}

template <class F> static float time_ms(F f, int iterations) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    f(); f();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iterations; i++) f();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / iterations;
}

static void errors(const std::vector<__nv_bfloat16> &got, const std::vector<float> &want,
                   const std::vector<int> &rows, int heads, double out[3]) {
    double num = 0, den = 0, worst = 0;
    for (size_t s = 0; s < rows.size(); s++)
        for (int h = 0; h < heads; h++)
            for (int d = 0; d < 128; d++) {
                double g = __bfloat162float(got[(size_t(rows[s]) * heads + h) * 128 + d]);
                double w = want[(s * heads + h) * 128 + d];
                num += (g - w) * (g - w); den += w * w;
                worst = fmax(worst, fabs(g - w));
            }
    out[0] = sqrt(num / den); out[1] = worst; out[2] = 0;
}

int main(int argc, char **argv) {
    int sequence = argc > 1 ? atoi(argv[1]) : 29565;
    int heads = argc > 2 ? atoi(argv[2]) : 56;
    size_t n = size_t(sequence) * heads * 128;
    float scale = 1.0f / sqrtf(128.0f);
    __nv_bfloat16 *q, *k, *v, *o2, *o3;
    float *lse, *ref;
    CHECK(cudaMalloc(&q, n * 2)); CHECK(cudaMalloc(&k, n * 2)); CHECK(cudaMalloc(&v, n * 2));
    CHECK(cudaMalloc(&o2, n * 2)); CHECK(cudaMalloc(&o3, n * 2));
    CHECK(cudaMalloc(&lse, size_t(heads) * sequence * 4));
    // Amplitudes give attention logits with a spread like trained models.
    fill<<<(n + 255) / 256, 256>>>(q, n, 1, 2.0f);
    fill<<<(n + 255) / 256, 256>>>(k, n, 2, 2.0f);
    fill<<<(n + 255) / 256, 256>>>(v, n, 3, 1.0f);
    CHECK(cudaDeviceSynchronize());
    auto fa2 = [&] { CHECK(h3_flash_bf16(o2, q, k, v, lse, 1, sequence, heads, heads, scale, false, 0)); };
    auto fa3 = [&] { CHECK(h3_flash3_bf16(o3, q, k, v, lse, 1, sequence, heads, heads, scale, 0)); };
    int iterations = sequence > 20000 ? 5 : 10;
    float t2 = time_ms(fa2, iterations), t3 = time_ms(fa3, iterations);
    double flops = 4.0 * double(sequence) * sequence * heads * 128;
    std::vector<int> rows;
    for (int i = 0; i < 64; i++) rows.push_back(int((long long)i * 7919 % sequence));
    rows.push_back(sequence - 1);
    int *rows_d;
    CHECK(cudaMalloc(&rows_d, rows.size() * 4));
    CHECK(cudaMalloc(&ref, rows.size() * heads * 128 * 4));
    CHECK(cudaMemcpy(rows_d, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaFuncSetAttribute(reference, cudaFuncAttributeMaxDynamicSharedMemorySize, sequence * 4));
    reference<<<dim3(rows.size(), heads), 256, sequence * 4>>>(q, k, v, rows_d, ref, sequence, heads, scale);
    CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> h2(n), h3(n);
    std::vector<float> want(rows.size() * heads * 128);
    CHECK(cudaMemcpy(h2.data(), o2, n * 2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h3.data(), o3, n * 2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(want.data(), ref, want.size() * 4, cudaMemcpyDeviceToHost));
    double e2[3], e3[3];
    errors(h2, want, rows, heads, e2);
    errors(h3, want, rows, heads, e3);
    double diff = 0;
    size_t differing = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(double(__bfloat162float(h2[i])) - __bfloat162float(h3[i]));
        diff = fmax(diff, d);
        differing += d != 0;
    }
    printf("{\"sequence\":%d,\"heads\":%d,\"fa2_ms\":%.3f,\"fa3_ms\":%.3f,"
           "\"fa2_tflops\":%.1f,\"fa3_tflops\":%.1f,\"speedup\":%.3f,"
           "\"fa2_rel_l2\":%.3e,\"fa2_max_abs\":%.3e,\"fa3_rel_l2\":%.3e,\"fa3_max_abs\":%.3e,"
           "\"fa2_vs_fa3_max_abs\":%.3e,\"fa2_vs_fa3_differing\":%.4f}\n",
           sequence, heads, t2, t3, flops / t2 / 1e9, flops / t3 / 1e9, t2 / t3,
           e2[0], e2[1], e3[0], e3[1], diff, double(differing) / n);
    return 0;
}
