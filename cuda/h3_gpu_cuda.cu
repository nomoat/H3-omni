// CUDA correctness backend. No PyTorch runtime and no quantized execution.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <sys/sysinfo.h>
#include <vector>
#include <new>
extern "C" {
#include "h3_cuda.h"
#include "h3_flash.h"
#include "h3_gpu.h"
#include "h3_metal.h"
}
#include "vae_attention.cuh"
using uint = unsigned int;
using ushort = unsigned short;
struct h3_gpu {
    cudaStream_t stream{};
    cudaStream_t copy_stream{};  // asynchronous readback (vae_fused.cuh)
    cudaEvent_t copy_ready{};
    cudaEvent_t start_event{}, stop_event{};
    bool timing_open{};
    cublasHandle_t blas{};
    void *lt_workspace{};
    cudnnHandle_t dnn{};
    h3_gpu_stats stats{};
    char error[512]{};
};
struct h3_gpu_tensor {
    void *data{};
    size_t elements{}, bytes{};
    h3_gpu_dtype dtype{};
    h3_gpu *owner{};
    void *tf32_low{};  // FP32 remainder after the TF32-exact data (3xTF32 weights)
};
static void h3_gpu_set_error(h3_gpu *g, const char *format, ...) {
    if (!g)
        return;
    va_list ap;
    va_start(ap, format);
    vsnprintf(g->error, sizeof(g->error), format, ap);
    va_end(ap);
}
static int cuda_ok(h3_gpu *g, cudaError_t e) {
    if (e == cudaSuccess)
        return 1;
    h3_gpu_set_error(g, "CUDA: %s", cudaGetErrorString(e));
    return 0;
}
static int blas_ok(h3_gpu *g, cublasStatus_t e) {
    if (e == CUBLAS_STATUS_SUCCESS)
        return 1;
    h3_gpu_set_error(g, "cuBLAS status %d", int(e));
    return 0;
}
static int dnn_ok(h3_gpu *g, cudnnStatus_t e) {
    if (e == CUDNN_STATUS_SUCCESS)
        return 1;
    h3_gpu_set_error(g, "cuDNN: %s", cudnnGetErrorString(e));
    return 0;
}
static int launch_ok(h3_gpu *g) {
    ++g->stats.direct_dispatches;
    return cuda_ok(g, cudaGetLastError());
}
static int h3_gpu_require_elements(h3_gpu *g, const h3_gpu_tensor *t, size_t n, const char *what) {
    if (t && t->owner == g && t->elements >= n)
        return 1;
    h3_gpu_set_error(g, "invalid tensor: %s (need %zu elements)", what, n);
    return 0;
}
static int h3_gpu_require_bf16(h3_gpu *g, const h3_gpu_tensor *t, size_t n, const char *what) {
    if (!h3_gpu_require_elements(g, t, n, what))
        return 0;
    if (t->dtype == H3_GPU_BF16)
        return 1;
    h3_gpu_set_error(g, "expected BF16: %s", what);
    return 0;
}
struct Binding {
    void *p[16]{};
    alignas(16) unsigned char values[16][128]{};
    void buffer(int slot, const h3_gpu_tensor *t, size_t offset) {
        p[slot] = t ? (char *)t->data + offset : nullptr;
    }
    void arg(int slot, const void *a, size_t n) {
        if (n > 128)
            abort();
        memcpy(values[slot], a, n);
    }
    template <class T> T get(int slot) const {
        T v;
        memcpy(&v, values[slot], sizeof(v));
        return v;
    }
};
#include "primitives.cuh"
#include "fast_primitives.cuh"
// Hand-written fast elementwise kernels (fast_primitives.cuh) are bit-identical
// to the generated ones. H3_GENERATED_ELEMENTWISE=1 selects the generated
// kernels for A/B comparisons.
static bool fast_elementwise() {
    static const bool fast = !getenv("H3_GENERATED_ELEMENTWISE");
    return fast;
}
static bool aligned_to(const void *p, size_t bytes) { return (uintptr_t)p % bytes == 0; }
static dim3 MTLSizeMake(size_t x, size_t y, size_t z) { return dim3(x, y, z); }
template <class F>
static int dispatch(h3_gpu *gpu, const char *name, dim3 grid, dim3 block, F bind) {
    Binding binding;
    bind(binding);
#include "dispatch.inc"
    h3_gpu_set_error(gpu, "unknown CUDA primitive: %s", name);
    return 0;
}
template <class F> static int h3_gpu_dispatch_1d(h3_gpu *g, const char *n, size_t x, F f) {
    return dispatch(g, n, dim3((x + 255) / 256), dim3(256), f);
}
template <class F>
static int h3_gpu_dispatch_2d(h3_gpu *g, const char *n, size_t x, size_t y, F f) {
    return dispatch(g, n, dim3((x + 31) / 32, (y + 7) / 8), dim3(32, 8), f);
}
template <class F> static int h3_gpu_dispatch_3d(h3_gpu *g, const char *n, dim3 s, F f) {
    // CUDA allows at most 65535 blocks in y and z (10 s audio has 320000
    // samples in y). Taller grids use taller blocks. Every 3D primitive
    // computes its own global index and checks bounds without shared memory,
    // so the block shape does not change results.
    unsigned by = 4, bz = 1;
    while ((s.y + by - 1) / by > 65535 && 32 * by * bz < 1024) by *= 2;
    while ((s.z + bz - 1) / bz > 65535 && 32 * by * bz < 1024) bz *= 2;
    if ((s.y + by - 1) / by > 65535 || (s.z + bz - 1) / bz > 65535) {
        h3_gpu_set_error(g, "%s: grid %u x %u x %u is too large", n, s.x, s.y, s.z);
        return 0;
    }
    return dispatch(g, n, dim3((s.x + 31) / 32, (s.y + by - 1) / by, (s.z + bz - 1) / bz),
                    dim3(32, by, bz), f);
}
template <class F> static int h3_gpu_dispatch_rows(h3_gpu *g, const char *n, size_t rows, F f) {
    return dispatch(g, n, dim3(rows), dim3(256), f);
}
// Without a release threshold the pool returns freed memory to the driver at
// every synchronization, and the next allocation pays for mapping it again.
static int keep_pool_memory(h3_gpu *g) {
    int device = 0;
    cudaMemPool_t pool{};
    uint64_t threshold = UINT64_MAX;
    return cuda_ok(g, cudaGetDevice(&device)) &&
           cuda_ok(g, cudaDeviceGetDefaultMemPool(&pool, device)) &&
           cuda_ok(g, cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
}
extern "C" h3_gpu *h3_gpu_create(const char *, char *error, size_t size) {
    h3_gpu *g = new h3_gpu;
    if (!cuda_ok(g, cudaStreamCreate(&g->stream)) || !blas_ok(g, cublasCreate(&g->blas)) ||
        !cuda_ok(g, cudaEventCreate(&g->start_event)) ||
        !cuda_ok(g, cudaEventCreate(&g->stop_event)) ||
        !blas_ok(g, cublasSetStream(g->blas, g->stream)) ||
        !blas_ok(g, cublasSetMathMode(g->blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) ||
        !dnn_ok(g, cudnnCreate(&g->dnn)) || !dnn_ok(g, cudnnSetStream(g->dnn, g->stream)) ||
        !keep_pool_memory(g)) {
        if (error && size)
            snprintf(error, size, "%s", g->error);
        h3_gpu_free(g);
        return nullptr;
    }
    return g;
}
extern "C" void h3_gpu_free(h3_gpu *g) {
    if (!g)
        return;
    if (g->stream)
        cudaStreamSynchronize(g->stream);
    if (g->copy_stream) {
        cudaStreamSynchronize(g->copy_stream);
        cudaStreamDestroy(g->copy_stream);
    }
    if (g->copy_ready)
        cudaEventDestroy(g->copy_ready);
    if (g->dnn)
        cudnnDestroy(g->dnn);
    if (g->lt_workspace)
        cudaFree(g->lt_workspace);
    if (g->blas)
        cublasDestroy(g->blas);
    if (g->start_event)
        cudaEventDestroy(g->start_event);
    if (g->stop_event)
        cudaEventDestroy(g->stop_event);
    if (g->stream)
        cudaStreamDestroy(g->stream);
    delete g;
}
extern "C" int h3_gpu_is_m5(const h3_gpu *) { return 0; }
extern "C" int h3_gpu_has_nax_mlp(const h3_gpu *) { return 0; }
extern "C" int h3_gpu_has_int8_mlp(const h3_gpu *) { return 0; }
static size_t item_size(h3_gpu_dtype d) {
    return d == H3_GPU_BF16 || d == H3_GPU_F16 ? 2 : d == H3_GPU_I8 ? 1 : 4;
}
// Device memory comes from the stream-ordered pool. cudaFree synchronizes the
// whole device, and per-operation temporaries made the GPU idle between
// kernels. The pool keeps freed memory for reuse; when it cannot grow, idle
// pool memory is returned to the driver and the allocation is retried.
// Addresses change, values do not.
static int device_alloc(h3_gpu *g, void **data, size_t bytes) {
    cudaError_t e = cudaMallocAsync(data, bytes, g->stream);
    if (e == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        cudaMemPool_t pool{};
        int device = 0;
        if (cudaDeviceSynchronize() == cudaSuccess && cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess)
            cudaMemPoolTrimTo(pool, 0);
        e = cudaMallocAsync(data, bytes, g->stream);
    }
    return cuda_ok(g, e);
}
static void device_free(h3_gpu *g, void *data) {
    if (data)
        cudaFreeAsync(data, g->stream);
}
static h3_gpu_tensor *allocate(h3_gpu *g, size_t n, h3_gpu_dtype d) {
    if (!g || n > SIZE_MAX / item_size(d))
        return nullptr;
    auto *t = new h3_gpu_tensor;
    t->elements = n;
    t->bytes = n * item_size(d);
    t->dtype = d;
    t->owner = g;
    if (!device_alloc(g, &t->data, std::max(t->bytes, size_t(1)))) {
        delete t;
        return nullptr;
    }
    g->stats.allocated_bytes += t->bytes;
    g->stats.live_bytes += t->bytes;
    g->stats.tensor_allocations++;
    g->stats.peak_live_bytes = std::max(g->stats.peak_live_bytes, g->stats.live_bytes);
    return t;
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *g, size_t n) {
    return allocate(g, n, H3_GPU_F32);
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *g, size_t n) {
    return allocate(g, n, H3_GPU_BF16);
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *g, size_t n) {
    return allocate(g, n, H3_GPU_I8);
}
extern "C" void h3_gpu_tensor_free(h3_gpu_tensor *t) {
    if (!t)
        return;
    device_free(t->owner, t->data);
    t->owner->stats.live_bytes -= t->bytes;
    if (t->tf32_low) {
        device_free(t->owner, t->tf32_low);
        t->owner->stats.live_bytes -= t->bytes;
    }
    delete t;
}
extern "C" size_t h3_gpu_tensor_elements(const h3_gpu_tensor *t) { return t ? t->elements : 0; }
extern "C" h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *t) {
    return t ? t->dtype : H3_GPU_F32;
}
static int transfer(const h3_gpu_tensor *t, void *host, size_t off, size_t n, h3_gpu_dtype d,
                    bool read) {
    if (!t || t->dtype != d || off > t->elements || n > t->elements - off || (!host && n))
        return 0;
    if (!cuda_ok(t->owner, cudaStreamSynchronize(t->owner->stream)))
        return 0;
    void *dev = (char *)t->data + off * item_size(d);
    return cuda_ok(t->owner, cudaMemcpy(read ? host : dev, read ? dev : host, n * item_size(d),
                                        read ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice));
}
#define TRANSFER(suffix, type, dtype)                                                              \
    extern "C" int h3_gpu_tensor_read_##suffix(const h3_gpu_tensor *t, type *v, size_t n) {        \
        return transfer(t, v, 0, n, dtype, true);                                                  \
    }                                                                                              \
    extern "C" int h3_gpu_tensor_write_##suffix(h3_gpu_tensor *t, const type *v, size_t n) {       \
        return transfer(t, (void *)v, 0, n, dtype, false);                                         \
    }                                                                                              \
    extern "C" int h3_gpu_tensor_write_##suffix##_range(h3_gpu_tensor *t, size_t o, const type *v, \
                                                        size_t n) {                                \
        return transfer(t, (void *)v, o, n, dtype, false);                                         \
    }                                                                                              \
    extern "C" h3_gpu_tensor *h3_gpu_tensor_from_##suffix(h3_gpu *g, const type *v, size_t n) {    \
        auto *t = allocate(g, n, dtype);                                                           \
        if (t && !transfer(t, (void *)v, 0, n, dtype, false)) {                                    \
            h3_gpu_tensor_free(t);                                                                 \
            return nullptr;                                                                        \
        }                                                                                          \
        return t;                                                                                  \
    }
TRANSFER(f32, float, H3_GPU_F32)
TRANSFER(bf16, uint16_t, H3_GPU_BF16)
extern "C" int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *t, size_t o, float *v, size_t n) {
    return transfer(t, v, o, n, H3_GPU_F32, true);
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *g, const uint32_t *v, size_t n) {
    auto *t = allocate(g, n, H3_GPU_U32);
    if (t && !transfer(t, (void *)v, 0, n, H3_GPU_U32, false)) {
        h3_gpu_tensor_free(t);
        return nullptr;
    }
    return t;
}
static int read_file(h3_gpu_tensor *t, const char *path, uint64_t offset, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        h3_gpu_set_error(t->owner, "cannot open %s", path);
        return 0;
    }
    if (fseeko(f, offset, SEEK_SET)) {
        fclose(f);
        return 0;
    }
    size_t total = n * item_size(t->dtype);
    std::vector<unsigned char> buf(std::min(total, size_t(16 << 20)));
    if (!cuda_ok(t->owner, cudaStreamSynchronize(t->owner->stream))) {
        fclose(f);
        return 0;
    }
    for (size_t pos = 0; pos < total;) {
        size_t count = std::min(buf.size(), total - pos);
        if (fread(buf.data(), 1, count, f) != count ||
            !cuda_ok(t->owner, cudaMemcpy((char *)t->data + pos, buf.data(), count,
                                          cudaMemcpyHostToDevice))) {
            fclose(f);
            return 0;
        }
        pos += count;
    }
    fclose(f);
    return 1;
}
static h3_gpu_tensor *load_file(h3_gpu *g, const char *p, uint64_t o, size_t n, h3_gpu_dtype d) {
    auto *t = allocate(g, n, d);
    if (t && !read_file(t, p, o, n)) {
        h3_gpu_tensor_free(t);
        return nullptr;
    }
    return t;
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *g, const char *p, uint64_t o, size_t n) {
    return load_file(g, p, o, n, H3_GPU_BF16);
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *g, const char *p, uint64_t o, size_t n) {
    return load_file(g, p, o, n, H3_GPU_F32);
}
extern "C" int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *t, const char *p, uint64_t o, size_t n,
                                            char *e, size_t z) {
    if (!t || t->dtype != H3_GPU_BF16 || n > t->elements)
        return 0;
    int ok = read_file(t, p, o, n);
    if (!ok && e && z)
        snprintf(e, z, "cannot read BF16 payload from %s", p);
    return ok;
}
extern "C" int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *t, const char *p, uint64_t o, size_t n,
                                              char *e, size_t z) {
    return h3_gpu_tensor_read_file_bf16(t, p, o, n, e, z);
}
extern "C" int h3_gpu_begin(h3_gpu *g) {
    if (!g)
        return 0;
    if (!g->timing_open) {
        if (!cuda_ok(g, cudaEventRecord(g->start_event, g->stream)))
            return 0;
        g->timing_open = true;
    }
    return 1;
}
extern "C" int h3_gpu_continue(h3_gpu *g) { return g && cuda_ok(g, cudaGetLastError()); }
extern "C" int h3_gpu_submit(h3_gpu *g) {
    if (!g)
        return 0;
    if (g->timing_open && !cuda_ok(g, cudaEventRecord(g->stop_event, g->stream)))
        return 0;
    auto a = std::chrono::steady_clock::now();
    int ok = cuda_ok(g, cudaStreamSynchronize(g->stream));
    if (ok && g->timing_open) {
        float milliseconds = 0;
        ok = cuda_ok(g, cudaEventElapsedTime(&milliseconds, g->start_event, g->stop_event));
        if (ok)
            g->stats.gpu_seconds += milliseconds / 1000.0;
        g->timing_open = false;
    }
    g->stats.command_wait_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    g->stats.submissions++;
    return ok;
}
extern "C" const char *h3_gpu_error(const h3_gpu *g) { return g ? g->error : "null CUDA context"; }
extern "C" int h3_gpu_get_stats(const h3_gpu *g, h3_gpu_stats *s) {
    if (!g || !s)
        return 0;
    *s = g->stats;
    return 1;
}
extern "C" void h3_gpu_profile_set_label(h3_gpu *, const char *) {}
extern "C" void h3_gpu_profile_mark(h3_gpu *g, const char *label) {
    if (getenv("H3_PROFILE")) {
        h3_gpu_submit(g);
        fprintf(stderr, "CUDA %s: live %.3f GiB; peak %.3f GiB\n", label,
                g->stats.live_bytes / 1073741824.0, g->stats.peak_live_bytes / 1073741824.0);
    }
}
extern "C" int h3_metal_probe(h3_device_info *i, char *e, size_t z) {
    cudaDeviceProp p;
    auto r = cudaGetDeviceProperties(&p, 0);
    if (r != cudaSuccess) {
        if (e && z)
            snprintf(e, z, "%s", cudaGetErrorString(r));
        return 0;
    }
    memset(i, 0, sizeof(*i));
    snprintf(i->name, sizeof(i->name), "%.127s", p.name);
    snprintf(i->architecture, sizeof(i->architecture), "CUDA sm_%d%d", p.major, p.minor);
    struct sysinfo s;
    sysinfo(&s);
    i->physical_memory = (uint64_t)s.totalram * s.mem_unit;
    i->recommended_working_set = p.totalGlobalMem;
    i->max_buffer_length = p.totalGlobalMem;
    return 1;
}
extern "C" {
#include "primitive_wrappers.inc"
}
// h3_gpu_gate_bf16 and h3_gpu_swiglu_bf16 are written by hand (the generator
// skips them) so they can use the vectorized kernels in fast_primitives.cuh.
// The checks and the fallback dispatch are the generated wrappers' code.
extern "C" int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual,
                                const h3_gpu_tensor *branch, const h3_gpu_tensor *modulation,
                                const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width,
                                uint32_t slots, uint32_t gate_slot) {
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, residual, count, "gate residual") ||
        !h3_gpu_require_bf16(gpu, branch, count, "gate branch") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, "gate modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "gate row map") ||
        row_map->dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, "gate output") ||
        gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    if (fast_elementwise() && count && width % 8 == 0 && aligned_to(residual->data, 16) &&
        aligned_to(branch->data, 16) && aligned_to(modulation->data, 16) &&
        aligned_to(output->data, 16)) {
        size_t threads = count / 8;
        h3_fast_gate_bf16<<<dim3((unsigned)((threads + 255) / 256)), dim3(256), 0, gpu->stream>>>(
            (const ushort *)residual->data, (const ushort *)branch->data,
            (const ushort *)modulation->data, (const uint *)row_map->data, (ushort *)output->data,
            args);
        return launch_ok(gpu);
    }
    return h3_gpu_dispatch_2d(gpu, "h3_gate_bf16", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, residual, 0);
            encoder.buffer(1, branch, 0);
            encoder.buffer(2, modulation, 0);
            encoder.buffer(3, row_map, 0);
            encoder.buffer(4, output, 0);
            encoder.arg(5, &args, sizeof(args));
        });
}
extern "C" int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *fused,
                                  uint32_t rows, uint32_t width) {
    if (!h3_gpu_require_bf16(gpu, fused, (size_t)rows * width * 2, "SwiGLU input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width, "SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    size_t count = (size_t)rows * width;
    if (fast_elementwise() && count && width % 8 == 0 && aligned_to(fused->data, 16) &&
        aligned_to(output->data, 16)) {
        size_t threads = count / 8;
        h3_fast_swiglu_bf16<<<dim3((unsigned)((threads + 255) / 256)), dim3(256), 0, gpu->stream>>>(
            (const ushort *)fused->data, (ushort *)output->data, args);
        return launch_ok(gpu);
    }
    return h3_gpu_dispatch_2d(gpu, "h3_swiglu_bf16", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, fused, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}
// Fused gate + AdaLN (gate = true) or AdaLN alone, see fast_primitives.cuh.
// Returns -1 when the shape or alignment does not fit; the caller then uses
// the generated kernels.
template <int C>
static void launch_fast_gate_adaln(h3_gpu *g, bool gate, ushort *res, const ushort *x,
                                   const ushort *branch, const ushort *gm, const ushort *w,
                                   const ushort *m, const uint *map, ushort *y,
                                   fast_gate_adaln_args a) {
    if (gate)
        h3_fast_gate_adaln_bf16<C, true><<<dim3(a.rows), dim3(128), 0, g->stream>>>(
            x, branch, gm, res, w, m, map, y, a);
    else
        h3_fast_gate_adaln_bf16<C, false><<<dim3(a.rows), dim3(128), 0, g->stream>>>(
            x, branch, gm, res, w, m, map, y, a);
}
static int fast_gate_adaln(h3_gpu *g, bool gate, ushort *res, const ushort *x, const ushort *branch,
                           const ushort *gm, const ushort *w, const ushort *m, const uint *map,
                           ushort *y, fast_gate_adaln_args a) {
    if (!fast_elementwise() || !a.rows || !a.width || a.width % 4 || a.width > 512 * 16)
        return -1;
    const void *ptrs[] = {res, x, branch, gm, w, m, y};
    for (const void *p : ptrs)
        if (!aligned_to(p, 8)) return -1;
    unsigned chunks = (a.width + 511) / 512;
    if (chunks <= 4) launch_fast_gate_adaln<4>(g, gate, res, x, branch, gm, w, m, map, y, a);
    else if (chunks <= 8) launch_fast_gate_adaln<8>(g, gate, res, x, branch, gm, w, m, map, y, a);
    else if (chunks <= 11) launch_fast_gate_adaln<11>(g, gate, res, x, branch, gm, w, m, map, y, a);
    else launch_fast_gate_adaln<16>(g, gate, res, x, branch, gm, w, m, map, y, a);
    return launch_ok(g);
}
// Qwen uses separate FP32 square, mean, rsqrt and multiply operations.
// Preserve Reduce.cuh's four accumulators and ascending shuffle order.
__global__ void qwen_norm(ushort *output, const ushort *input, const ushort *weight,
                          uint width, float epsilon, bool vectorized) {
    __shared__ float sums[512];
    uint row = blockIdx.x, tid = threadIdx.x;
    const ushort *x = input + size_t(row) * width;
    float partial[4] = {0,0,0,0};
    if (vectorized) {
        for (uint start = 4 * tid; start < width; start += 4 * blockDim.x)
            for (uint i = 0; i < 4; ++i) {
                float value = h3_bf16_to_f32(x[start+i]);
                partial[i] += value * value;
            }
    } else {
        for (uint start = tid; start < width; start += 4 * blockDim.x)
            for (uint i = 0; i < 4; ++i) {
                uint index = start + i * blockDim.x;
                if (index < width) {
                    float value = h3_bf16_to_f32(x[index]);
                    partial[i] += value * value;
                }
            }
    }
    float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    sums[tid] = sum;
    for (uint distance = blockDim.x / 2; distance >= 32; distance >>= 1) {
        __syncthreads();
        if (tid < distance) sums[tid] = sum = sum + sums[tid + distance];
    }
    __syncthreads();
    for (uint distance = 1; distance < 32; distance <<= 1)
        sum += __shfl_down_sync(0xffffffff, sum, distance);
    if (tid == 0) sums[0] = sum;
    __syncthreads();
    float inverse = rsqrtf(sums[0] * (1.0f / float(width)) + epsilon);
    for (uint column = tid; column < width; column += blockDim.x) {
        float normalized = bf16_round(h3_bf16_to_f32(x[column]) * inverse);
        output[size_t(row) * width + column] = h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
    }
}
extern "C" int h3_gpu_qwen_rms_norm_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                         const h3_gpu_tensor *w, uint rows, uint width, float eps) {
    if (!h3_gpu_require_bf16(g, x, size_t(rows) * width, "Qwen RMSNorm input") ||
        !h3_gpu_require_bf16(g, y, size_t(rows) * width, "Qwen RMSNorm output") ||
        !h3_gpu_require_bf16(g, w, width, "Qwen RMSNorm weight"))
        return 0;
    if (width == 128 || width == 5120) {
        uint output_power = 1;
        while (output_power * 2 <= std::min(rows, 16u)) output_power *= 2;
        uint threads = std::min(width == 128 ? 128u : 512u, 512u / output_power);
        qwen_norm<<<rows, threads, 0, g->stream>>>(static_cast<ushort *>(y->data),
            static_cast<const ushort *>(x->data), static_cast<const ushort *>(w->data),
            width, eps, width > 128);
        return launch_ok(g);
    }
    norm_args args{rows, width, eps};
    return h3_gpu_dispatch_rows(g, "h3_qwen_rms_norm_bf16", rows, [&](Binding &b) {
        b.buffer(0, x, 0);
        b.buffer(1, w, 0);
        b.buffer(2, y, 0);
        b.arg(3, &args, sizeof(args));
    });
}
extern "C" int h3_gpu_head_rms_norm_bf16(h3_gpu *g, h3_gpu_tensor *tensor,
                                       const h3_gpu_tensor *weight, uint sequence,
                                       uint heads, uint dim, float epsilon) {
    if (dim == 128)
        return h3_gpu_qwen_rms_norm_bf16(g, tensor, tensor, weight, sequence * heads, dim, epsilon);
    if (!h3_gpu_require_bf16(g, tensor, size_t(sequence) * heads * dim, "head norm") ||
        !h3_gpu_require_bf16(g, weight, dim, "head norm weight")) return 0;
    head_norm_args args{sequence, heads, dim, epsilon};
    return h3_gpu_dispatch_2d(g, "h3_head_rms_norm_bf16", sequence, heads, [&](Binding &b) {
        b.buffer(0, tensor, 0); b.buffer(1, weight, 0); b.arg(2, &args, sizeof(args));
    });
}
static int copy_tensor(h3_gpu *g, h3_gpu_tensor *d, size_t od, const h3_gpu_tensor *s, size_t os,
                       size_t n, h3_gpu_dtype dt) {
    if (!d || !s || d->dtype != dt || s->dtype != dt || od > d->elements || n > d->elements - od ||
        os > s->elements || n > s->elements - os)
        return 0;
    g->stats.blit_copies++;
    return cuda_ok(g, cudaMemcpyAsync((char *)d->data + od * item_size(dt),
                                      (char *)s->data + os * item_size(dt), n * item_size(dt),
                                      cudaMemcpyDeviceToDevice, g->stream));
}
extern "C" int h3_gpu_copy_f32(h3_gpu *g, h3_gpu_tensor *d, size_t od, const h3_gpu_tensor *s,
                               size_t os, size_t n) {
    return copy_tensor(g, d, od, s, os, n, H3_GPU_F32);
}
extern "C" int h3_gpu_copy_bf16(h3_gpu *g, h3_gpu_tensor *d, size_t od, const h3_gpu_tensor *s,
                                size_t os, size_t n) {
    return copy_tensor(g, d, od, s, os, n, H3_GPU_BF16);
}
__global__ void bias_f32(float *y, const float *b, size_t n, int width) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += b[i % width];
}
__global__ void bias_bf16(ushort *y, const ushort *b, size_t n, int width) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = h3_f32_to_bf16(h3_bf16_to_f32(y[i]) + h3_bf16_to_f32(b[i % width]));
}
__global__ void bias_cast_bf16(ushort *y, const float *x, const ushort *bias, size_t count,
                               uint width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = h3_f32_to_bf16(x[i] + h3_bf16_to_f32(bias[i % width]));
}
static int linear_bias_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                           const h3_gpu_tensor *w, const h3_gpu_tensor *bias,
                           uint rows, uint k, uint n) {
    // Match PyTorch addmm's cuBLASLt bias epilogue and 1 MiB workspace.
    constexpr size_t workspace_bytes = 1024 * 1024;
    if (!g->lt_workspace && !cuda_ok(g, cudaMalloc(&g->lt_workspace, workspace_bytes))) return 0;
    cublasLtMatmulDesc_t descriptor{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulPreference_t preference{};
    cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    int ok = blas_ok(g, cublasLtMatmulDescCreate(&descriptor, CUBLAS_COMPUTE_32F, CUDA_R_32F)) &&
        blas_ok(g, cublasLtMatmulDescSetAttribute(descriptor, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose))) &&
        blas_ok(g, cublasLtMatmulDescSetAttribute(descriptor, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal))) &&
        blas_ok(g, cublasLtMatmulDescSetAttribute(descriptor, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) &&
        blas_ok(g, cublasLtMatmulDescSetAttribute(descriptor, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias->data, sizeof(bias->data))) &&
        blas_ok(g, cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, k, n, k)) &&
        blas_ok(g, cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, k, rows, k)) &&
        blas_ok(g, cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, rows, n)) &&
        blas_ok(g, cublasLtMatmulPreferenceCreate(&preference)) &&
        blas_ok(g, cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
    const void *pointers[] = {w->data, x->data, y->data, bias->data};
    const cublasLtMatmulPreferenceAttributes_t attributes[] = {
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    for (int i = 0; i < 4 && ok; ++i) {
        uint32_t alignment = 256;
        while (reinterpret_cast<uintptr_t>(pointers[i]) % alignment) alignment /= 2;
        ok = blas_ok(g, cublasLtMatmulPreferenceSetAttribute(preference, attributes[i], &alignment, sizeof(alignment)));
    }
    cublasLtMatmulHeuristicResult_t algorithm{};
    int count = 0;
    auto handle = reinterpret_cast<cublasLtHandle_t>(g->blas);
    if (ok) ok = blas_ok(g, cublasLtMatmulAlgoGetHeuristic(handle, descriptor, a, b, c, c,
                                                         preference, 1, &algorithm, &count));
    if (ok && !count) { h3_gpu_set_error(g, "No cuBLASLt FP32 bias algorithm"); ok = 0; }
    float alpha = 1, beta = 0;
    if (ok) ok = blas_ok(g, cublasLtMatmul(handle, descriptor, &alpha, w->data, a, x->data, b,
        &beta, y->data, c, y->data, c, &algorithm.algo, g->lt_workspace, workspace_bytes, g->stream));
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (descriptor) cublasLtMatmulDescDestroy(descriptor);
    ++g->stats.mps_linear_dispatches;
    return ok;
}
__global__ void split_tf32(const float *x, float *high, float *low, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x[i]));
    float h = __uint_as_float(bits);
    high[i] = h;
    low[i] = x[i] - h;
}
extern "C" int h3_gpu_tensor_split_tf32(h3_gpu *g, h3_gpu_tensor *w) {
    if (!h3_gpu_require_elements(g, w, w ? w->elements : 0, "TF32 split weight") ||
        w->dtype != H3_GPU_F32 || w->tf32_low) return 0;
    if (!device_alloc(g, &w->tf32_low, w->bytes)) return 0;
    g->stats.live_bytes += w->bytes;
    g->stats.peak_live_bytes = std::max(g->stats.peak_live_bytes, g->stats.live_bytes);
    split_tf32<<<(w->elements + 255) / 256, 256, 0, g->stream>>>(
        (const float *)w->data, (float *)w->data, (float *)w->tf32_low, w->elements);
    return launch_ok(g);
}
// Three Hopper TF32 products xh*wl + xl*wh + xh*wh accumulate into y.
static int gemm_3xtf32(h3_gpu *g, h3_gpu_tensor *y, const float *high, const float *low,
                       const h3_gpu_tensor *w, uint rows, uint k, uint n) {
    const void *weights[3] = {w->tf32_low, w->data, w->data};
    const float *inputs[3] = {high, low, high};
    float one = 1, zero = 0;
    int ok = 1;
    for (int term = 0; ok && term < 3; term++)
        ok = blas_ok(g, cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one,
                                     weights[term], CUDA_R_32F, k, inputs[term], CUDA_R_32F, k,
                                     term ? &one : &zero, y->data, CUDA_R_32F, n,
                                     CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    return ok;
}
// y = x w^T + b with 3xTF32: the activation is split per call, the weight at
// load. Three Hopper TF32 products accumulate into y, then the bias is added.
static int linear_3xtf32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                         const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint rows, uint k,
                         uint n) {
    size_t count = size_t(rows) * k;
    auto *parts = allocate(g, 2 * count, H3_GPU_F32);
    if (!parts) return 0;
    float *high = (float *)parts->data, *low = high + count;
    split_tf32<<<(count + 255) / 256, 256, 0, g->stream>>>((const float *)x->data, high, low,
                                                           count);
    int ok = launch_ok(g) && gemm_3xtf32(g, y, high, low, w, rows, k, n);
    if (ok && b) {
        ok = h3_gpu_require_elements(g, b, n, "linear bias") && b->dtype == H3_GPU_F32;
        if (ok) {
            bias_f32<<<(size_t(rows) * n + 255) / 256, 256, 0, g->stream>>>(
                (float *)y->data, (float *)b->data, size_t(rows) * n, n);
            ok = launch_ok(g);
        }
    }
    h3_gpu_tensor_free(parts);
    ++g->stats.mps_linear_dispatches;
    return ok;
}
static int linear(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x, const h3_gpu_tensor *w,
                  const h3_gpu_tensor *b, uint rows, uint k, uint n, h3_gpu_dtype dtype) {
    if (!rows || !k || !n || rows > INT_MAX || k > INT_MAX || n > INT_MAX ||
        !h3_gpu_require_elements(g, x, (size_t)rows * k, "linear input") ||
        !h3_gpu_require_elements(g, w, (size_t)n * k, "linear weight") ||
        !h3_gpu_require_elements(g, y, (size_t)rows * n, "linear output") || x->dtype != dtype ||
        w->dtype != dtype || y->dtype != dtype)
        return 0;
    if (dtype == H3_GPU_F32 && w->tf32_low)
        return linear_3xtf32(g, y, x, w, b, rows, k, n);
    if (dtype == H3_GPU_F32 && b && k > 1 && n > 1) {
        if (!h3_gpu_require_elements(g, b, n, "linear bias") || b->dtype != dtype) return 0;
        return linear_bias_f32(g, y, x, w, b, rows, k, n);
    }
    float alpha = 1, beta = 0;
    cudaDataType_t dt = dtype == H3_GPU_F32 ? CUDA_R_32F : CUDA_R_16BF;
    if (dtype == H3_GPU_BF16 && b) {
        if (!h3_gpu_require_bf16(g, b, n, "linear bias"))
            return 0;
        auto *acc = allocate(g, size_t(rows) * n, H3_GPU_F32);
        if (!acc)
            return 0;
        int ok =
            blas_ok(g, cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, w->data,
                                    dt, k, x->data, dt, k, &beta, acc->data, CUDA_R_32F, n,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        if (ok) {
            bias_cast_bf16<<<(size_t(rows) * n + 255) / 256, 256, 0, g->stream>>>(
                (ushort *)y->data, (float *)acc->data, (ushort *)b->data, size_t(rows) * n, n);
            ok = launch_ok(g);
        }
        h3_gpu_tensor_free(acc);
        ++g->stats.mps_linear_dispatches;
        return ok;
    }
    if (!blas_ok(g, cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, w->data, dt,
                                 k, x->data, dt, k, &beta, y->data, dt, n,
                                 CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT)))
        return 0;
    ++g->stats.mps_linear_dispatches;
    if (b) {
        if (!h3_gpu_require_elements(g, b, n, "linear bias") || b->dtype != dtype)
            return 0;
        if (dtype == H3_GPU_F32)
            bias_f32<<<((size_t)rows * n + 255) / 256, 256, 0, g->stream>>>(
                (float *)y->data, (float *)b->data, (size_t)rows * n, n);
        else
            bias_bf16<<<((size_t)rows * n + 255) / 256, 256, 0, g->stream>>>(
                (ushort *)y->data, (ushort *)b->data, (size_t)rows * n, n);
    }
    return cuda_ok(g, cudaGetLastError());
}
extern "C" int h3_gpu_linear_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                 const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint r, uint k,
                                 uint n) {
    return linear(g, y, x, w, b, r, k, n, H3_GPU_F32);
}
__global__ void time_features(float *output, const float *times, uint rows, uint width) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(rows) * width) return;
    uint half = width / 2, column = index % width;
    float exponent = (-9.210340371976184f * float(column % half)) / float(half);
    float angle = times[index / width] * expf(exponent);
    output[index] = column < half ? cosf(angle) : sinf(angle);
}
extern "C" int h3_gpu_time_features_f32(h3_gpu *g, h3_gpu_tensor *y,
                                      const h3_gpu_tensor *times, uint rows, uint width) {
    if (!rows || !width || width % 2 ||
        !h3_gpu_require_elements(g, times, rows, "timesteps") ||
        !h3_gpu_require_elements(g, y, size_t(rows) * width, "timestep features") ||
        times->dtype != H3_GPU_F32 || y->dtype != H3_GPU_F32) return 0;
    time_features<<<(size_t(rows) * width + 255) / 256, 256, 0, g->stream>>>(
        static_cast<float *>(y->data), static_cast<float *>(times->data), rows, width);
    return launch_ok(g);
}
static int time_linear(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                       const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                       uint rows, uint k, uint n, const uint32_t *row_counts,
                       uint group_count, h3_gpu_dtype dtype) {
    if (!rows || !row_counts || !group_count || group_count > rows) {
        h3_gpu_set_error(g, "invalid timestep projection groups");
        return 0;
    }
    uint covered = 0;
    for (uint group = 0; group < group_count; ++group) {
        uint count = row_counts[group];
        if (!count || count > rows - covered) {
            h3_gpu_set_error(g, "invalid timestep projection row count");
            return 0;
        }
        covered += count;
    }
    if (covered != rows) {
        h3_gpu_set_error(g, "timestep projection groups do not cover all rows");
        return 0;
    }
    if (!h3_gpu_require_elements(g, x, size_t(rows) * k, "timestep input") ||
        !h3_gpu_require_elements(g, y, size_t(rows) * n, "timestep output")) return 0;
    uint row = 0;
    for (uint group = 0; group < group_count; ++group) {
        uint count = row_counts[group];
        auto input = *x, output = *y;
        input.data = static_cast<char *>(x->data) + size_t(row) * k * item_size(dtype);
        output.data = static_cast<char *>(y->data) + size_t(row) * n * item_size(dtype);
        input.elements = size_t(count) * k;
        output.elements = size_t(count) * n;
        input.bytes = input.elements * item_size(dtype);
        output.bytes = output.elements * item_size(dtype);
        if (!linear(g, &output, &input, w, b, count, k, n, dtype)) return 0;
        row += count;
    }
    return 1;
}
extern "C" int h3_gpu_time_linear_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                    const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                                    uint rows, uint k, uint n,
                                    const uint32_t *row_counts, uint group_count) {
    return time_linear(g, y, x, w, b, rows, k, n, row_counts, group_count, H3_GPU_F32);
}
extern "C" int h3_gpu_time_linear_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                     const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                                     uint rows, uint k, uint n,
                                     const uint32_t *row_counts, uint group_count) {
    return time_linear(g, y, x, w, b, rows, k, n, row_counts, group_count, H3_GPU_BF16);
}
extern "C" int h3_gpu_linear_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                  const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint r, uint k,
                                  uint n) {
    return linear(g, y, x, w, b, r, k, n, H3_GPU_BF16);
}
__global__ void timestep_row_map(uint *output, const uint *input, uint rows) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows) output[index] = input[index] / 3;
}
extern "C" int h3_gpu_final_heads_f32(h3_gpu *g, h3_gpu_tensor *video_output,
                         h3_gpu_tensor *audio_output, const h3_gpu_tensor *hidden,
                         const h3_gpu_tensor *norm, const h3_gpu_tensor *modulation,
                         const h3_gpu_tensor *row_map, const h3_gpu_tensor *video_weight,
                         const h3_gpu_tensor *video_bias, const h3_gpu_tensor *audio_weight,
                         const h3_gpu_tensor *audio_bias, uint rows, uint width,
                         uint video_start, uint video_rows, uint audio_start, uint audio_rows) {
    // Project the entire packed sequence, then select each modality, as in PyTorch.
    auto *map = allocate(g, rows, H3_GPU_U32);
    auto *normalized = allocate(g, size_t(rows) * width, H3_GPU_BF16);
    auto *f32 = allocate(g, size_t(rows) * width, H3_GPU_F32);
    auto *video = allocate(g, size_t(rows) * 96, H3_GPU_F32);
    auto *audio = allocate(g, size_t(rows) * 32, H3_GPU_F32);
    int ok = map && normalized && f32 && video && audio;
    if (ok) {
        timestep_row_map<<<(rows+255)/256,256,0,g->stream>>>(static_cast<uint *>(map->data),
            static_cast<const uint *>(row_map->data), rows);
        ok = launch_ok(g) &&
            h3_gpu_adaln_bf16(g, normalized, hidden, norm, modulation, map, rows, width, 2, 0, 1, 1e-5f) &&
            h3_gpu_cast_bf16_to_f32(g, f32, normalized, rows * width) &&
            h3_gpu_linear_f32(g, video, f32, video_weight, video_bias, rows, width, 96) &&
            h3_gpu_linear_f32(g, audio, f32, audio_weight, audio_bias, rows, width, 32) &&
            h3_gpu_copy_f32(g, video_output, 0, video, size_t(video_start) * 96, size_t(video_rows) * 96) &&
            h3_gpu_copy_f32(g, audio_output, 0, audio, size_t(audio_start) * 32, size_t(audio_rows) * 32);
    }
    for (auto *tensor : {map, normalized, f32, video, audio}) h3_gpu_tensor_free(tensor);
    return ok;
}
extern "C" int h3_gpu_patch_linear_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                        const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint r,
                                        uint k, uint n) {
    auto *t = allocate(g, (size_t)r * n, H3_GPU_F32);
    if (!t)
        return 0;
    int ok = h3_gpu_linear_f32(g, t, x, w, b, r, k, n) && h3_gpu_cast_f32_to_bf16(g, y, t, r * n);
    h3_gpu_tensor_free(t);
    return ok;
}
extern "C" int h3_gpu_patch_linear_bf16_offset(h3_gpu *g, h3_gpu_tensor *y, size_t yo,
                                               const h3_gpu_tensor *x, size_t xo,
                                               const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                                               uint r, uint k, uint n) {
    if (xo > x->elements || yo > y->elements)
        return 0;
    auto xx = *x, yy = *y;
    xx.data = (float *)x->data + xo;
    xx.elements -= xo;
    yy.data = (ushort *)y->data + yo;
    yy.elements -= yo;
    return h3_gpu_patch_linear_bf16(g, &yy, &xx, w, b, r, k, n);
}
__global__ void scatter_rows(ushort *y, const ushort *x, const uint *map, uint rows, uint width) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * width)
        y[(size_t)map[i / width] * width + i % width] = x[i];
}
extern "C" int h3_gpu_patch_linear_bf16_map(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                            const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                                            const h3_gpu_tensor *map, uint outrows, uint r, uint k,
                                            uint n) {
    auto *t = allocate(g, (size_t)r * n, H3_GPU_BF16);
    if (!t)
        return 0;
    int ok = h3_gpu_patch_linear_bf16(g, t, x, w, b, r, k, n);
    if (ok) {
        scatter_rows<<<((size_t)r * n + 255) / 256, 256, 0, g->stream>>>(
            (ushort *)y->data, (ushort *)t->data, (uint *)map->data, r, n);
        ok = launch_ok(g);
    }
    h3_gpu_tensor_free(t);
    return ok;
}
extern "C" int h3_gpu_mlp_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                               const h3_gpu_tensor *w1, const h3_gpu_tensor *w2, uint rows,
                               uint dim, uint hidden, uint outdim) {
    auto *fc1 = allocate(g, (size_t)rows * hidden * 2, H3_GPU_BF16);
    auto *act = allocate(g, (size_t)rows * hidden, H3_GPU_BF16);
    int ok = fc1 && act && h3_gpu_linear_bf16(g, fc1, x, w1, nullptr, rows, dim, 2 * hidden) &&
             h3_gpu_swiglu_bf16(g, act, fc1, rows, hidden) &&
             h3_gpu_linear_bf16(g, y, act, w2, nullptr, rows, hidden, outdim);
    h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(act);
    return ok;
}
extern "C" int h3_gpu_adaln_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                 const h3_gpu_tensor *w, const h3_gpu_tensor *m,
                                 const h3_gpu_tensor *map, uint r, uint width, uint slots, uint sh,
                                 uint sc, float eps) {
    adaln_args a{r, width, slots, sh, sc, eps};
    if (x && w && m && map && y) {
        int fast = fast_gate_adaln(g, false, nullptr, (const ushort *)x->data, nullptr, nullptr,
                                   (const ushort *)w->data, (const ushort *)m->data,
                                   (const uint *)map->data, (ushort *)y->data,
                                   fast_gate_adaln_args{r, width, slots, 0, sh, sc, eps});
        if (fast >= 0) return fast;
    }
    return h3_gpu_dispatch_rows(g, "h3_adaln_bf16", r, [&](Binding &b) {
        b.buffer(0, x, 0);
        b.buffer(1, w, 0);
        b.buffer(2, m, 0);
        b.buffer(3, map, 0);
        b.buffer(4, y, 0);
        b.arg(5, &a, sizeof(a));
    });
}
extern "C" int h3_gpu_adaln_bf16_offset(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                        size_t off, const h3_gpu_tensor *w, const h3_gpu_tensor *m,
                                        const h3_gpu_tensor *map, uint r, uint width, uint slots,
                                        uint sh, uint sc, float eps) {
    if (off > x->elements)
        return 0;
    auto xx = *x;
    xx.data = (ushort *)x->data + off;
    xx.elements -= off;
    return h3_gpu_adaln_bf16(g, y, &xx, w, m, map, r, width, slots, sh, sc, eps);
}
static int qkv_rope(h3_gpu *g, h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                    const h3_gpu_tensor *x, const h3_gpu_tensor *qn, const h3_gpu_tensor *kn,
                    const h3_gpu_tensor *co, const h3_gpu_tensor *si, uint seq, uint heads,
                    uint dim, uint half, float eps, uint grouped) {
    qkv_args a{seq, heads, dim, half, grouped, eps};
    size_t items = (size_t)seq * heads;
    if (fast_elementwise() && dim == 128 && half % 4 == 0 && half * 2 <= dim && items && q && k &&
        v && x && qn && kn && co && si && aligned_to(co->data, 8) && aligned_to(si->data, 8) &&
        aligned_to(x->data, 8) && aligned_to(qn->data, 8) && aligned_to(kn->data, 8) &&
        aligned_to(q->data, 8) && aligned_to(k->data, 8) && aligned_to(v->data, 8)) {
        h3_fast_qkv_rope_bf16<<<dim3((unsigned)((items + 7) / 8)), dim3(256), 0, g->stream>>>(
            (const ushort *)x->data, (const ushort *)qn->data, (const ushort *)kn->data,
            (const ushort *)co->data, (const ushort *)si->data, (ushort *)q->data,
            (ushort *)k->data, (ushort *)v->data, a);
        return launch_ok(g);
    }
    return h3_gpu_dispatch_3d(g, "h3_qkv_rope_bf16", dim3(dim, heads, seq), [&](Binding &b) {
        b.buffer(0, x, 0);
        b.buffer(1, qn, 0);
        b.buffer(2, kn, 0);
        b.buffer(3, co, 0);
        b.buffer(4, si, 0);
        b.buffer(5, q, 0);
        b.buffer(6, k, 0);
        b.buffer(7, v, 0);
        b.arg(8, &a, sizeof(a));
    });
}
extern "C" int h3_gpu_qkv_rope_bf16(h3_gpu *g, h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                                    const h3_gpu_tensor *x, const h3_gpu_tensor *qn,
                                    const h3_gpu_tensor *kn, const h3_gpu_tensor *co,
                                    const h3_gpu_tensor *si, uint s, uint h, uint d, uint rh,
                                    float eps) {
    return qkv_rope(g, q, k, v, x, qn, kn, co, si, s, h, d, rh, eps, 0);
}
extern "C" int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *g, h3_gpu_tensor *q, h3_gpu_tensor *k,
                                            h3_gpu_tensor *v, const h3_gpu_tensor *x,
                                            const h3_gpu_tensor *qn, const h3_gpu_tensor *kn,
                                            const h3_gpu_tensor *co, const h3_gpu_tensor *si,
                                            uint s, uint h, uint d, uint rh, float eps) {
    return qkv_rope(g, q, k, v, x, qn, kn, co, si, s, h, d, rh, eps, 1);
}
extern "C" int h3_gpu_euler_bf16(h3_gpu *g, h3_gpu_tensor *sample, size_t off,
                                 const h3_gpu_tensor *v, const h3_gpu_tensor *old, uint n,
                                 float delta, float ratio) {
    h3_euler_args a{uint(off), n, delta, ratio};
    return h3_gpu_dispatch_1d(g, "h3_euler_bf16", n, [&](Binding &b) {
        b.buffer(0, sample, 0);
        b.buffer(1, v, 0);
        b.buffer(2, old ? old : v, 0);
        b.arg(3, &a, sizeof(a));
    });
}

template <class T> __device__ float as_float(T x) { return float(x); }
template <> __device__ float as_float<ushort>(ushort x) { return h3_bf16_to_f32(x); }
template <class T> __device__ T from_float(float x) { return T(x); }
template <> __device__ ushort from_float<ushort>(float x) { return h3_f32_to_bf16(x); }
template <class T>
__global__ void pack_heads(float *y, const T *x, size_t total, int n, int h, int kh, int d) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int dim = i % d, row = i / d % n, head = i / (d * n) % h, batch = i / (size_t(d) * n * h);
    int sourcehead = head / (h / kh);
    y[i] = as_float(x[((size_t(batch) * n + row) * kh + sourcehead) * d + dim]);
}
template <class T>
__global__ void unpack_heads(T *y, const float *x, size_t total, int n, int h, int d) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int dim = i % d, head = i / d % h, row = i / (d * h) % n, batch = i / (size_t(d) * h * n);
    y[i] = from_float<T>(x[((size_t(batch) * h + head) * n + row) * d + dim]);
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
        if (threadIdx.x < s)
            tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x], tmp[threadIdx.x + s]);
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
        if (threadIdx.x < s)
            tmp[threadIdx.x] += tmp[threadIdx.x + s];
        __syncthreads();
    }
    sum = tmp[0];
    for (int k = threadIdx.x; k < n; k += blockDim.x)
        v[k] /= sum;
}
static bool use_flash3() {
    static const bool enabled = [] {
        const char *value = getenv("H3_FLASH3");
        return !value || strcmp(value, "0");
    }();
    return enabled;
}
static int sdpa(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q, const h3_gpu_tensor *k,
                const h3_gpu_tensor *v, uint batch, uint n, uint h, uint kh, uint d, float scale,
                bool causal, bool allow_flash3 = true) {
    if (!n || !h || !kh || h % kh || !d || !batch)
        return 0;
    if (q->dtype == H3_GPU_BF16 && d == 128) {
        auto *lse = allocate(g, size_t(batch) * h * n, H3_GPU_F32);
        if (!lse)
            return 0;
        // Non-causal (DiT joint) attention uses the FlashAttention-3 Hopper
        // kernel; H3_FLASH3=0 selects the FlashAttention-2 reference kernel.
        int ok = cuda_ok(g, !causal && allow_flash3 && use_flash3()
            ? h3_flash3_bf16(y->data, q->data, k->data, v->data,
                             static_cast<float *>(lse->data), batch, n, h, kh, scale,
                             g->stream)
            : h3_flash_bf16(y->data, q->data, k->data, v->data,
                            static_cast<float *>(lse->data), batch, n, h, kh,
                            scale, causal, g->stream));
        h3_gpu_tensor_free(lse);
        ++g->stats.mps_sdpa_dispatches;
        return ok;
    }
    size_t size = (size_t)batch * h * n * d;
    uint chunk = std::min(n, 256u), bh = batch * h;
    auto *buf = allocate(g, size * 4 + (size_t)bh * chunk * n, H3_GPU_F32);
    if (!buf)
        return 0;
    float *qq = (float *)buf->data, *kk = qq + size, *vv = kk + size, *out = vv + size,
          *p = out + size;
    if (q->dtype == H3_GPU_F32) {
        pack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>(qq, (float *)q->data, size, n, h, h,
                                                              d);
        pack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>(kk, (float *)k->data, size, n, h, kh,
                                                              d);
        pack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>(vv, (float *)v->data, size, n, h, kh,
                                                              d);
    } else {
        pack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>(qq, (ushort *)q->data, size, n, h, h,
                                                              d);
        pack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>(kk, (ushort *)k->data, size, n, h, kh,
                                                              d);
        pack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>(vv, (ushort *)v->data, size, n, h, kh,
                                                              d);
    }
    float zero = 0, one = 1;
    int ok = launch_ok(g);
    for (uint start = 0; ok && start < n; start += chunk) {
        uint count = std::min(chunk, n - start);
        long long stride = (long long)n * d, ps = (long long)n * count;
        ok = blas_ok(g, cublasSgemmStridedBatched(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, count, d,
                                                  &scale, kk, d, stride, qq + (size_t)start * d, d,
                                                  stride, &zero, p, n, ps, bh));
        softmax_rows<<<bh * count, 256, 0, g->stream>>>(p, n, count, start, causal);
        ok = ok && launch_ok(g);
        if (ok)
            ok = blas_ok(g, cublasSgemmStridedBatched(g->blas, CUBLAS_OP_N, CUBLAS_OP_N, d, count,
                                                      n, &one, vv, d, stride, p, n, ps, &zero,
                                                      out + (size_t)start * d, d, stride, bh));
    }
    if (ok) {
        if (y->dtype == H3_GPU_F32)
            unpack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>((float *)y->data, out, size, n,
                                                                    h, d);
        else
            unpack_heads<<<(size + 255) / 256, 256, 0, g->stream>>>((ushort *)y->data, out, size, n,
                                                                    h, d);
        ok = launch_ok(g);
    }
    h3_gpu_tensor_free(buf);
    ++g->stats.mps_sdpa_dispatches;
    return ok;
}
extern "C" int h3_gpu_sdpa_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q,
                               const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint n, uint h,
                               uint d, float s) {
    return sdpa(g, y, q, k, v, 1, n, h, h, d, s, false);
}
extern "C" int h3_gpu_sdpa_3xtf32_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q,
                                      const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint n,
                                      uint h, uint d, float s) {
    size_t size = size_t(n) * h * d;
    if (!n || !h || d != vae_attention::D || size_t(n) * h > INT_MAX) {
        h3_gpu_set_error(g, "3xTF32 attention needs head dim 64");
        return 0;
    }
    if (!h3_gpu_require_elements(g, q, size, "attention query") ||
        !h3_gpu_require_elements(g, k, size, "attention key") ||
        !h3_gpu_require_elements(g, v, size, "attention value") ||
        !h3_gpu_require_elements(g, y, size, "attention output"))
        return 0;
    if (q->dtype != H3_GPU_F32 || k->dtype != H3_GPU_F32 || v->dtype != H3_GPU_F32 ||
        y->dtype != H3_GPU_F32) {
        h3_gpu_set_error(g, "3xTF32 attention needs FP32 tensors");
        return 0;
    }
    size_t words = vae_attention::split_words(int(n), int(h));
    auto *split = allocate(g, 2 * words, H3_GPU_F32);
    if (!split) return 0;
    unsigned *split_k = (unsigned *)split->data, *split_v = split_k + words;
    int ok = cuda_ok(g, vae_attention::launch((float *)y->data, (const float *)q->data,
                                              (const float *)k->data, (const float *)v->data,
                                              split_k, split_v, int(n), int(h), s, g->stream));
    h3_gpu_tensor_free(split);
    g->stats.direct_dispatches += 2;
    ++g->stats.mps_sdpa_dispatches;
    return ok;
}
extern "C" int h3_gpu_sdpa_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q,
                                const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint n, uint h,
                                uint d, float s) {
    return sdpa(g, y, q, k, v, 1, n, h, h, d, s, false);
}
extern "C" int h3_gpu_sdpa_fa2_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q,
                                    const h3_gpu_tensor *k, const h3_gpu_tensor *v,
                                    uint n, uint h, uint d, float s) {
    return sdpa(g, y, q, k, v, 1, n, h, h, d, s, false, false);
}
extern "C" int h3_gpu_sdpa_causal_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q,
                                      const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint b,
                                      uint n, uint h, uint d, float s) {
    return sdpa(g, y, q, k, v, b, n, h, h, d, s, true);
}
extern "C" int h3_gpu_gqa_causal_bf16(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *q,
                                      const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint n,
                                      uint h, uint kh, uint d, float s) {
    return sdpa(g, y, q, k, v, 1, n, h, kh, d, s, true);
}

// Channel-last activations with original PyTorch OI.../IO... filter storage.
__global__ void channel_order(float *output, const float *input, size_t count, int channels,
                              int spatial, bool to_first) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    size_t channel = i % channels, position = i / channels % spatial;
    size_t batch = i / (size_t(channels) * spatial);
    size_t first = (batch * channels + channel) * spatial + position;
    if (to_first)
        output[first] = input[i];
    else
        output[i] = input[first];
}
static int convolution(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x, const h3_gpu_tensor *w,
                       const h3_gpu_tensor *b, int batch, int ci, int co, std::vector<int> spatial,
                       std::vector<int> kernel, std::vector<int> stride, std::vector<int> pad,
                       std::vector<int> dilation, bool transpose) {
    int rank = spatial.size(), nd = rank + 2;
    std::vector<int> xd{batch, ci}, yd{batch, co}, wd{transpose ? ci : co, transpose ? co : ci};
    for (int a = 0; a < rank; a++) {
        if (spatial[a] <= 0 || kernel[a] <= 0 || stride[a] <= 0)
            return 0;
        xd.push_back(spatial[a]);
        yd.push_back(
            transpose
                ? (spatial[a] - 1) * stride[a] - 2 * pad[a] + dilation[a] * (kernel[a] - 1) + 1
                : (spatial[a] + 2 * pad[a] - dilation[a] * (kernel[a] - 1) - 1) / stride[a] + 1);
        wd.push_back(kernel[a]);
    }
    auto strides = [](const std::vector<int> &dims) {
        std::vector<int> s(dims.size());
        s[1] = 1;
        int step = dims[1];
        for (int a = int(dims.size()) - 1; a >= 2; --a) {
            s[a] = step;
            step *= dims[a];
        }
        s[0] = step;
        return s;
    };
    auto xs = strides(xd), ys = strides(yd);
    size_t nx = (size_t)xs[0] * batch, ny = (size_t)ys[0] * batch, nw = 1;
    for (int v : wd)
        nw *= v;
    if (!h3_gpu_require_elements(g, x, nx, "conv input") ||
        !h3_gpu_require_elements(g, y, ny, "conv output") ||
        !h3_gpu_require_elements(g, w, nw, "conv weight") || x->dtype != H3_GPU_F32 ||
        y->dtype != H3_GPU_F32 || w->dtype != H3_GPU_F32)
        return 0;
    h3_gpu_tensor *channel_first = nullptr;
    const void *input_data = x->data;
    void *output_data = y->data;
    if (transpose) {
        channel_first = allocate(g, nx + ny, H3_GPU_F32);
        if (!channel_first)
            return 0;
        input_data = channel_first->data;
        output_data = (float *)channel_first->data + nx;
        channel_order<<<(nx + 255) / 256, 256, 0, g->stream>>>(
            (float *)channel_first->data, (float *)x->data, nx, ci, nx / (batch * ci), true);
        xs[nd - 1] = ys[nd - 1] = 1;
        for (int axis = nd - 2; axis >= 0; --axis) {
            xs[axis] = xs[axis + 1] * xd[axis + 1];
            ys[axis] = ys[axis + 1] * yd[axis + 1];
        }
    }
    cudnnTensorDescriptor_t dx{}, dy{};
    cudnnFilterDescriptor_t dw{};
    cudnnConvolutionDescriptor_t dc{};
    void *workspace = nullptr;
    size_t ws = 0;
    int ok = dnn_ok(g, cudnnCreateTensorDescriptor(&dx)) &&
             dnn_ok(g, cudnnCreateTensorDescriptor(&dy)) &&
             dnn_ok(g, cudnnCreateFilterDescriptor(&dw)) &&
             dnn_ok(g, cudnnCreateConvolutionDescriptor(&dc));
    if (ok)
        ok =
            dnn_ok(g, cudnnSetTensorNdDescriptor(dx, CUDNN_DATA_FLOAT, nd, xd.data(), xs.data())) &&
            dnn_ok(g, cudnnSetTensorNdDescriptor(dy, CUDNN_DATA_FLOAT, nd, yd.data(), ys.data())) &&
            dnn_ok(g, cudnnSetFilterNdDescriptor(dw, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nd,
                                                 wd.data())) &&
            dnn_ok(g, cudnnSetConvolutionNdDescriptor(dc, rank, pad.data(), stride.data(),
                                                      dilation.data(), CUDNN_CROSS_CORRELATION,
                                                      CUDNN_DATA_FLOAT)) &&
            dnn_ok(g, cudnnSetConvolutionMathType(dc, CUDNN_FMA_MATH));
    if (ok) {
        if (transpose)
            ok = dnn_ok(g, cudnnGetConvolutionBackwardDataWorkspaceSize(
                               g->dnn, dw, dx, dc, dy, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &ws));
        else
            ok = dnn_ok(g,
                        cudnnGetConvolutionForwardWorkspaceSize(
                            g->dnn, dx, dw, dc, dy, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &ws));
    }
    if (ok && ws)
        ok = device_alloc(g, &workspace, ws);
    float one = 1, zero = 0;
    if (ok) {
        if (transpose)
            ok = dnn_ok(g, cudnnConvolutionBackwardData(g->dnn, &one, dw, w->data, dx, input_data,
                                                        dc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                                        workspace, ws, &zero, dy, output_data));
        else
            ok = dnn_ok(g, cudnnConvolutionForward(g->dnn, &one, dx, x->data, dw, w->data, dc,
                                                   CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                   workspace, ws, &zero, dy, y->data));
    }
    if (ok && transpose) {
        channel_order<<<(ny + 255) / 256, 256, 0, g->stream>>>(
            (float *)y->data, (float *)output_data, ny, co, ny / (batch * co), false);
        ok = launch_ok(g);
    }
    if (ok && b) {
        bias_f32<<<(ny + 255) / 256, 256, 0, g->stream>>>((float *)y->data, (float *)b->data, ny,
                                                          co);
        ok = launch_ok(g);
    }
    device_free(g, workspace);
    h3_gpu_tensor_free(channel_first);
    if (dc)
        cudnnDestroyConvolutionDescriptor(dc);
    if (dw)
        cudnnDestroyFilterDescriptor(dw);
    if (dx)
        cudnnDestroyTensorDescriptor(dx);
    if (dy)
        cudnnDestroyTensorDescriptor(dy);
    g->stats.mps_conv_dispatches++;
    return ok;
}
extern "C" int h3_gpu_conv1d_stride_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                        const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint bs,
                                        uint len, uint ci, uint co, uint k, uint s, uint p,
                                        uint d) {
    return convolution(g, y, x, w, b, bs, ci, co, {1, int(len)}, {1, int(k)}, {1, int(s)},
                       {0, int(p)}, {1, int(d)}, false);
}
extern "C" int h3_gpu_conv1d_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                 const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint bs, uint len,
                                 uint ci, uint co, uint k, uint p, uint d) {
    return h3_gpu_conv1d_stride_f32(g, y, x, w, b, bs, len, ci, co, k, 1, p, d);
}
extern "C" int h3_gpu_conv_transpose1d_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                           const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint bs,
                                           uint len, uint ci, uint co, uint k, uint s, uint p) {
    return convolution(g, y, x, w, b, bs, ci, co, {1, int(len)}, {1, int(k)}, {1, int(s)},
                       {0, int(p)}, {1, 1}, true);
}
extern "C" int h3_gpu_conv3d_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                 const h3_gpu_tensor *w, const h3_gpu_tensor *b, uint bs,
                                 uint depth, uint height, uint width, uint ci, uint co, uint kd,
                                 uint kh, uint kw, uint sd, uint sh, uint sw) {
    return convolution(g, y, x, w, b, bs, ci, co, {int(depth), int(height), int(width)},
                       {int(kd), int(kh), int(kw)}, {int(sd), int(sh), int(sw)}, {0, 0, 0},
                       {1, 1, 1}, false);
}
extern "C" int h3_gpu_grouped_qkv_linear_rope_bf16(h3_gpu *g, h3_gpu_tensor *q, h3_gpu_tensor *k,
                                                   h3_gpu_tensor *v, h3_gpu_tensor *qkv,
                                                   const h3_gpu_tensor *x, const h3_gpu_tensor *w,
                                                   const h3_gpu_tensor *qn, const h3_gpu_tensor *kn,
                                                   const h3_gpu_tensor *co, const h3_gpu_tensor *si,
                                                   uint rows, uint in, uint heads, uint dim,
                                                   uint half, float eps) {
    return h3_gpu_linear_bf16(g, qkv, x, w, nullptr, rows, in, heads * dim * 3) &&
           h3_gpu_grouped_qkv_rope_bf16(g, q, k, v, qkv, qn, kn, co, si, rows, heads, dim, half,
                                        eps);
}
extern "C" int h3_gpu_gate_adaln_bf16(h3_gpu *g, h3_gpu_tensor *res, h3_gpu_tensor *y,
                                      const h3_gpu_tensor *x, const h3_gpu_tensor *branch,
                                      const h3_gpu_tensor *w, const h3_gpu_tensor *gm,
                                      const h3_gpu_tensor *nm, const h3_gpu_tensor *map, uint r,
                                      uint width, uint slots, uint gate, uint sh, uint sc,
                                      float eps) {
    size_t count = (size_t)r * width;
    // Same checks as h3_gpu_gate_bf16; if one fails, the unfused calls below
    // report the error.
    if (fast_elementwise() && w && nm && y &&
        h3_gpu_require_bf16(g, x, count, "gate residual") &&
        h3_gpu_require_bf16(g, branch, count, "gate branch") &&
        h3_gpu_require_bf16(g, gm, 1, "gate modulation") &&
        h3_gpu_require_elements(g, map, r, "gate row map") && map->dtype == H3_GPU_U32 &&
        h3_gpu_require_bf16(g, res, count, "gate output") && gate < slots) {
        int fast = fast_gate_adaln(g, true, (ushort *)res->data, (const ushort *)x->data,
                                   (const ushort *)branch->data, (const ushort *)gm->data,
                                   (const ushort *)w->data, (const ushort *)nm->data,
                                   (const uint *)map->data, (ushort *)y->data,
                                   fast_gate_adaln_args{r, width, slots, gate, sh, sc, eps});
        if (fast >= 0) return fast;
    }
    return h3_gpu_gate_bf16(g, res, x, branch, gm, map, r, width, slots, gate) &&
           h3_gpu_adaln_bf16(g, y, res, w, nm, map, r, width, slots, sh, sc, eps);
}
extern "C" int h3_gpu_adaln_linear_bf16(h3_gpu *g, h3_gpu_tensor *y, h3_gpu_tensor *,
                                        const h3_gpu_tensor *x, size_t off, const h3_gpu_tensor *w,
                                        const h3_gpu_tensor *m, const h3_gpu_tensor *map,
                                        const h3_gpu_tensor *proj, const h3_gpu_tensor *b, uint r,
                                        uint width, uint out, uint slots, uint sh, uint sc,
                                        float eps) {
    auto *t = allocate(g, (size_t)r * width, H3_GPU_BF16);
    if (!t)
        return 0;
    int ok = h3_gpu_adaln_bf16_offset(g, t, x, off, w, m, map, r, width, slots, sh, sc, eps) &&
             h3_gpu_linear_bf16(g, y, t, proj, b, r, width, out);
    h3_gpu_tensor_free(t);
    return ok;
}
#include "vae_fused.cuh"
#include "vae_fp16.cuh"
#include "parallel.cuh"
#ifdef H3_VSA
#include "vsa.cuh"
#else
extern "C" h3_gpu_vsa *h3_gpu_vsa_new(h3_gpu *g, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned) {
    h3_gpu_set_error(g, "VSA requires a CUDA build with VSA=1");
    return nullptr;
}
extern "C" void h3_gpu_vsa_free(h3_gpu_vsa *) {}
extern "C" int h3_gpu_vsa_bf16(h3_gpu_vsa *, h3_gpu_tensor *, const h3_gpu_tensor *,
                                 const h3_gpu_tensor *, const h3_gpu_tensor *, const h3_gpu_tensor *) { return 0; }
#endif
#include "unsupported.inc"
