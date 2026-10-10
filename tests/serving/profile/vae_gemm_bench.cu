// Video-VAE linear layers y = x W^T (x [M,K] row-major, W [N,K] row-major):
// cuBLAS FP32 SIMT (current), cuBLAS 1xTF32, and CUTLASS 3xTF32
// (OpMultiplyAddFastF32). Speed and error against an FP64 reference.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cutlass/gemm/device/gemm.h"

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CHECK_BLAS(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "%s:%d cublas %d\n", __FILE__, __LINE__, int(s_)); exit(1); } } while (0)

using Gemm3xTF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 4, 4, false,
    cutlass::arch::OpMultiplyAddFastF32>;

__global__ void fill(float *x, size_t n, unsigned seed, float amplitude) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    unsigned h = unsigned(i) * 2654435761u ^ seed * 2246822519u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    x[i] = (float(h) / 4294967296.0f * 2.0f - 1.0f) * amplitude;
}

// Split x into a TF32-exact high part and the FP32 remainder.
__global__ void split_tf32(const float *x, float *high, float *low, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    unsigned bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x[i]));
    float h = __uint_as_float(bits);
    high[i] = h;
    low[i] = x[i] - h;
}

__global__ void widen(const float *x, double *y, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = x[i];
}

template <class F> static float time_ms(F f) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) f();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < 20; i++) f();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / 20;
}

static void error(const float *got_d, const std::vector<double> &want, size_t n, double *rel,
                  double *worst) {
    std::vector<float> got(n);
    CHECK(cudaMemcpy(got.data(), got_d, n * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, w = 0;
    for (size_t i = 0; i < n; i++) {
        double d = got[i] - want[i];
        num += d * d; den += want[i] * want[i];
        w = fmax(w, fabs(d) / (fabs(want[i]) + 1e-3));
    }
    *rel = sqrt(num / den); *worst = w;
}

int main() {
    const int shapes[][3] = {{1797, 6144, 2048}, {1797, 2048, 2048},
                             {1797, 16384, 2048}, {1797, 2048, 8192}};
    const char *names[] = {"qkv", "attn_out", "ff_w1", "ff_w2"};
    cublasHandle_t blas;
    CHECK_BLAS(cublasCreate(&blas));
    CHECK_BLAS(cublasSetMathMode(blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    for (int s = 0; s < 4; s++) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        size_t nx = size_t(M) * K, nw = size_t(N) * K, ny = size_t(M) * N;
        float *x, *w, *y;
        double *xd, *wd, *yd;
        CHECK(cudaMalloc(&x, nx * 4)); CHECK(cudaMalloc(&w, nw * 4)); CHECK(cudaMalloc(&y, ny * 4));
        CHECK(cudaMalloc(&xd, nx * 8)); CHECK(cudaMalloc(&wd, nw * 8)); CHECK(cudaMalloc(&yd, ny * 8));
        // Normalized activations and weights scaled like a trained linear layer.
        fill<<<(nx + 255) / 256, 256>>>(x, nx, 11 + s, 1.7f);
        fill<<<(nw + 255) / 256, 256>>>(w, nw, 23 + s, 1.7f / sqrtf(float(K)));
        widen<<<(nx + 255) / 256, 256>>>(x, xd, nx);
        widen<<<(nw + 255) / 256, 256>>>(w, wd, nw);
        // Column-major view: y^T [N,M] = W [N,K] (as K x N col-major, transposed) * x^T.
        double one_d = 1, zero_d = 0;
        CHECK_BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one_d, wd, K, xd, K,
                               &zero_d, yd, N));
        std::vector<double> want(ny);
        CHECK(cudaMemcpy(want.data(), yd, ny * 8, cudaMemcpyDeviceToHost));
        float one = 1, zero = 0;
        auto sgemm = [&] {
            CHECK_BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, CUDA_R_32F, K,
                                    x, CUDA_R_32F, K, &zero, y, CUDA_R_32F, N,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        auto tf32 = [&] {
            CHECK_BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, CUDA_R_32F, K,
                                    x, CUDA_R_32F, K, &zero, y, CUDA_R_32F, N,
                                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
        };
        Gemm3xTF32 gemm;
        // Row-major y [M,N] = x [M,K] * B, B = W^T as K x N column-major (ld K).
        Gemm3xTF32::Arguments args({M, N, K}, {x, K}, {w, K}, {y, N}, {y, N}, {1.0f, 0.0f});
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass cannot implement\n"); return 1; }
        auto x3 = [&] {
            if (gemm(args) != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass failed\n"); exit(1); }
        };
        // 3xTF32 from three Hopper TF32 cuBLAS products:
        // y = xh*wh + xh*wl + xl*wh, weights split once, activations per call.
        float *xh, *xl, *wh, *wl;
        CHECK(cudaMalloc(&xh, nx * 4)); CHECK(cudaMalloc(&xl, nx * 4));
        CHECK(cudaMalloc(&wh, nw * 4)); CHECK(cudaMalloc(&wl, nw * 4));
        split_tf32<<<(nw + 255) / 256, 256>>>(w, wh, wl, nw);
        auto split3 = [&] {
            split_tf32<<<(nx + 255) / 256, 256>>>(x, xh, xl, nx);
            const float *as[3] = {wl, wh, wh}, *bs[3] = {xh, xl, xh};
            for (int term = 0; term < 3; term++) {
                const float *beta = term ? &one : &zero;
                CHECK_BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, as[term],
                                        CUDA_R_32F, K, bs[term], CUDA_R_32F, K, beta, y,
                                        CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32,
                                        CUBLAS_GEMM_DEFAULT));
            }
        };
        double flops = 2.0 * M * N * K;
        double rel[4], worst[4];
        float t[4];
        t[0] = time_ms(sgemm); error(y, want, ny, &rel[0], &worst[0]);
        t[1] = time_ms(tf32);  error(y, want, ny, &rel[1], &worst[1]);
        t[2] = time_ms(x3);    error(y, want, ny, &rel[2], &worst[2]);
        t[3] = time_ms(split3); error(y, want, ny, &rel[3], &worst[3]);
        printf("{\"layer\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,"
               "\"sgemm_ms\":%.3f,\"sgemm_tflops\":%.1f,\"sgemm_rel_l2\":%.2e,\"sgemm_max_rel\":%.2e,"
               "\"tf32_ms\":%.3f,\"tf32_tflops\":%.1f,\"tf32_rel_l2\":%.2e,\"tf32_max_rel\":%.2e,"
               "\"x3tf32_ms\":%.3f,\"x3tf32_tflops\":%.1f,\"x3tf32_rel_l2\":%.2e,\"x3tf32_max_rel\":%.2e,"
               "\"split3_ms\":%.3f,\"split3_tflops\":%.1f,\"split3_rel_l2\":%.2e,\"split3_max_rel\":%.2e}\n",
               names[s], M, N, K, t[0], flops / t[0] / 1e9, rel[0], worst[0],
               t[1], flops / t[1] / 1e9, rel[1], worst[1], t[2], flops / t[2] / 1e9, rel[2], worst[2], t[3], flops / t[3] / 1e9, rel[3], worst[3]);
        cudaFree(xh); cudaFree(xl); cudaFree(wh); cudaFree(wl);
        cudaFree(x); cudaFree(w); cudaFree(y); cudaFree(xd); cudaFree(wd); cudaFree(yd);
    }
    return 0;
}
