// Video-VAE attention (1797 tokens, 32 heads x 64, FP32): the current FP32
// path (pack heads, cuBLAS SGEMM per 256-row query chunk, softmax_rows,
// unpack) versus the fused 3xTF32 kernel in cuda/vae_attention.cuh (and plain
// TF32 for comparison). Error against an FP64 reference on sampled rows.
// Usage: vae_attn_bench [TOKENS [HEADS]]
#include "vae_attention.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <vector>

using uint = unsigned;
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define BLAS(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "%s:%d cuBLAS %d\n", __FILE__, __LINE__, int(s_)); exit(1); } } while (0)

// Copies of the current backend kernels (cuda/h3_gpu_cuda.cu, FP32 branch of sdpa).
__global__ void pack_heads(float *y, const float *x, size_t total, int n, int h, int kh, int d) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int dim = i % d, row = i / d % n, head = i / (d * n) % h, batch = i / (size_t(d) * n * h);
    int sourcehead = head / (h / kh);
    y[i] = x[((size_t(batch) * n + row) * kh + sourcehead) * d + dim];
}
__global__ void unpack_heads(float *y, const float *x, size_t total, int n, int h, int d) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int dim = i % d, head = i / d % h, row = i / (d * h) % n, batch = i / (size_t(d) * h * n);
    y[i] = x[((size_t(batch) * h + head) * n + row) * d + dim];
}
__global__ void softmax_rows(float *x, int n, int qn, int start, bool causal) {
    int row = blockIdx.x;
    float *v = x + (size_t)row * n;
    int q = row % qn + start;
    __shared__ float tmp[256];
    float m = -INFINITY;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float a = causal && k > q ? -INFINITY : v[k];
        v[k] = a;
        m = fmaxf(m, a);
    }
    tmp[threadIdx.x] = m;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x], tmp[threadIdx.x + s]);
        __syncthreads();
    }
    m = tmp[0];
    __syncthreads();
    float sum = 0;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float a = expf(v[k] - m);
        v[k] = a;
        sum += a;
    }
    tmp[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) tmp[threadIdx.x] += tmp[threadIdx.x + s];
        __syncthreads();
    }
    sum = tmp[0];
    for (int k = threadIdx.x; k < n; k += blockDim.x) v[k] /= sum;
}

__global__ void fill(float *x, size_t n, unsigned seed, float amplitude) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    unsigned h = unsigned(i) * 2654435761u ^ seed * 2246822519u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    x[i] = (float(h) / 4294967296.0f * 2.0f - 1.0f) * amplitude;
}

// One block per (sampled row, head): softmax(q k^T * scale) v in FP64.
__global__ void reference(const float *q, const float *k, const float *v, const int *rows,
                          double *out, int n, int heads, double scale) {
    extern __shared__ double scores[];
    int sample = blockIdx.x, head = blockIdx.y, row = rows[sample];
    const float *qr = q + (size_t(row) * heads + head) * 64;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float *kr = k + (size_t(j) * heads + head) * 64;
        double s = 0;
        for (int d = 0; d < 64; d++) s += double(qr[d]) * kr[d];
        scores[j] = s * scale;
    }
    __syncthreads();
    __shared__ double maximum, total;
    if (threadIdx.x == 0) {
        double m = -INFINITY, t = 0;
        for (int j = 0; j < n; j++) m = fmax(m, scores[j]);
        for (int j = 0; j < n; j++) t += exp(scores[j] - m);
        maximum = m; total = t;
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        double acc = 0;
        for (int j = 0; j < n; j++)
            acc += exp(scores[j] - maximum) * v[(size_t(j) * heads + head) * 64 + threadIdx.x];
        out[(size_t(sample) * heads + head) * 64 + threadIdx.x] = acc / total;
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

struct Error { double rel_l2, max_abs, max_rel_row; };
static Error errors(const std::vector<float> &got, const std::vector<double> &want,
                    const std::vector<int> &rows, int heads) {
    double num = 0, den = 0, worst = 0, worst_row = 0;
    for (size_t s = 0; s < rows.size(); s++)
        for (int h = 0; h < heads; h++) {
            double rn = 0, rd = 0;
            for (int d = 0; d < 64; d++) {
                double g = got[(size_t(rows[s]) * heads + h) * 64 + d];
                double w = want[(s * heads + h) * 64 + d];
                num += (g - w) * (g - w); den += w * w;
                rn += (g - w) * (g - w); rd += w * w;
                worst = fmax(worst, fabs(g - w));
            }
            worst_row = fmax(worst_row, sqrt(rn / rd));
        }
    return {sqrt(num / den), worst, worst_row};
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1797, heads = argc > 2 ? atoi(argv[2]) : 32;
    const int d = 64;
    size_t size = size_t(n) * heads * d;
    float scale = 1.0f / sqrtf(64.0f);
    uint chunk = std::min(n, 256), bh = heads;
    float *q, *k, *v, *y_fp32, *y_3x, *y_1x, *buf;
    CHECK(cudaMalloc(&q, size * 4)); CHECK(cudaMalloc(&k, size * 4)); CHECK(cudaMalloc(&v, size * 4));
    CHECK(cudaMalloc(&y_fp32, size * 4)); CHECK(cudaMalloc(&y_3x, size * 4));
    CHECK(cudaMalloc(&y_1x, size * 4));
    unsigned *sk, *sv;
    CHECK(cudaMalloc(&sk, vae_attention::split_words(n, heads) * 4));
    CHECK(cudaMalloc(&sv, vae_attention::split_words(n, heads) * 4));
    CHECK(cudaMalloc(&buf, (size * 4 + size_t(bh) * chunk * n) * 4));
    cublasHandle_t blas;
    BLAS(cublasCreate(&blas));
    float *qq = buf, *kk = qq + size, *vv = kk + size, *out = vv + size, *p = out + size;
    auto current = [&] {
        pack_heads<<<(size + 255) / 256, 256>>>(qq, q, size, n, heads, heads, d);
        pack_heads<<<(size + 255) / 256, 256>>>(kk, k, size, n, heads, heads, d);
        pack_heads<<<(size + 255) / 256, 256>>>(vv, v, size, n, heads, heads, d);
        float zero = 0, one = 1;
        for (uint start = 0; start < uint(n); start += chunk) {
            uint count = std::min(chunk, uint(n) - start);
            long long stride = (long long)n * d, ps = (long long)n * count;
            BLAS(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, count, d, &scale, kk,
                                           d, stride, qq + (size_t)start * d, d, stride, &zero, p,
                                           n, ps, bh));
            softmax_rows<<<bh * count, 256>>>(p, n, count, start, false);
            BLAS(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, d, count, n, &one, vv,
                                           d, stride, p, n, ps, &zero, out + (size_t)start * d,
                                           d, stride, bh));
        }
        unpack_heads<<<(size + 255) / 256, 256>>>(y_fp32, out, size, n, heads, d);
        CHECK(cudaGetLastError());
    };
    auto fused3 = [&] { CHECK((vae_attention::launch<3, true>(y_3x, q, k, v, sk, sv, n, heads, scale, 0))); };
    auto fused1 = [&] { CHECK((vae_attention::launch<1, true>(y_1x, q, k, v, sk, sv, n, heads, scale, 0))); };
    // Variants: no promotion (accuracy study), and the v5 instruction order (must give identical bits).
    auto fused3np = [&] { CHECK((vae_attention::launch<3, false>(y_1x, q, k, v, sk, sv, n, heads, scale, 0))); };
    auto fused4 = [&] { CHECK((vae_attention::launch<3, true, 8, false>(y_1x, q, k, v, sk, sv, n, heads, scale, 0))); };

    std::vector<int> rows;
    for (int i = 0; i < 96; i++) rows.push_back(int((long long)i * 7919 % n));
    rows.push_back(n - 1);
    int *rows_d;
    double *ref;
    CHECK(cudaMalloc(&rows_d, rows.size() * 4));
    CHECK(cudaMalloc(&ref, rows.size() * heads * 64 * 8));
    CHECK(cudaMemcpy(rows_d, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaFuncSetAttribute(reference, cudaFuncAttributeMaxDynamicSharedMemorySize, n * 8));
    double flops = 4.0 * double(n) * n * heads * d;
    // Amplitude a gives logits with standard deviation a^2 / 3 (uniform inputs).
    const float amplitudes[] = {1.0f, 1.7320508f, 3.4641016f, 5.0f};
    for (float amplitude : amplitudes) {
        fill<<<(size + 255) / 256, 256>>>(q, size, 1, amplitude);
        fill<<<(size + 255) / 256, 256>>>(k, size, 2, amplitude);
        fill<<<(size + 255) / 256, 256>>>(v, size, 3, 1.0f);
        CHECK(cudaDeviceSynchronize());
        float t_cur = time_ms(current, 20), t_3x = time_ms(fused3, 20), t_1x = time_ms(fused1, 20);
        reference<<<dim3(rows.size(), heads), 256, n * 8>>>(q, k, v, rows_d, ref, n, heads, scale);
        CHECK(cudaDeviceSynchronize());
        std::vector<float> a(size), b(size), c(size);
        std::vector<double> want(rows.size() * heads * 64);
        CHECK(cudaMemcpy(a.data(), y_fp32, size * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), y_3x, size * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(c.data(), y_1x, size * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(want.data(), ref, want.size() * 8, cudaMemcpyDeviceToHost));
        Error ea = errors(a, want, rows, heads), eb = errors(b, want, rows, heads),
              ec = errors(c, want, rows, heads);
        float t_np = time_ms(fused3np, 20);
        CHECK(cudaMemcpy(c.data(), y_1x, size * 4, cudaMemcpyDeviceToHost));
        Error enp = errors(c, want, rows, heads);
        float t_4 = time_ms(fused4, 20);
        CHECK(cudaMemcpy(c.data(), y_1x, size * 4, cudaMemcpyDeviceToHost));
        Error e4 = errors(c, want, rows, heads);
        size_t order_differs = 0;
        for (size_t i = 0; i < size; i++) order_differs += memcmp(&b[i], &c[i], 4) != 0;
        printf("{\"interleave_vs_v5order_differing\":%zu}\n", order_differs);
        printf("{\"logit_std\":%.2f,\"fused3x_nopromote_ms\":%.3f,\"fused3x_nopromote_rel_l2\":%.3e,"
               "\"fused3x_nopromote_max_abs\":%.3e,\"fused3x_v5order_ms\":%.3f,\"fused3x_v5order_rel_l2\":%.3e,"
               "\"fused3x_v5order_max_abs\":%.3e}\n", amplitude * amplitude / 3, t_np, enp.rel_l2,
               enp.max_abs, t_4, e4.rel_l2, e4.max_abs);
        double num = 0, den = 0, worst = 0;
        for (size_t i = 0; i < size; i++) {
            double e = double(b[i]) - a[i];
            num += e * e; den += double(a[i]) * a[i]; worst = fmax(worst, fabs(e));
        }
        printf("{\"tokens\":%d,\"heads\":%d,\"logit_std\":%.2f,"
               "\"fp32_ms\":%.3f,\"fused3x_ms\":%.3f,\"fused1x_ms\":%.3f,"
               "\"fp32_tflops\":%.1f,\"fused3x_tflops\":%.1f,\"speedup\":%.2f,"
               "\"fp32_rel_l2\":%.3e,\"fp32_max_abs\":%.3e,\"fp32_worst_row\":%.3e,"
               "\"fused3x_rel_l2\":%.3e,\"fused3x_max_abs\":%.3e,\"fused3x_worst_row\":%.3e,"
               "\"tf32_rel_l2\":%.3e,\"tf32_max_abs\":%.3e,"
               "\"fused3x_vs_fp32_rel_l2\":%.3e,\"fused3x_vs_fp32_max_abs\":%.3e}\n",
               n, heads, amplitude * amplitude / 3, t_cur, t_3x, t_1x, flops / t_cur / 1e9,
               flops / t_3x / 1e9, t_cur / t_3x, ea.rel_l2, ea.max_abs, ea.max_rel_row,
               eb.rel_l2, eb.max_abs, eb.max_rel_row, ec.rel_l2, ec.max_abs, sqrt(num / den),
               worst);
    }
    return 0;
}
