// Video VAE serving helpers: asynchronous tile readback and 3xTF32 fusions.
// Included at the end of h3_gpu_cuda.cu (needs its allocator and helpers).
//
// Fusions keep every rounding step of the unfused path. The build uses
// --fmad=false, so a fused kernel that adds the bias with __fadd_rn on load
// computes the same FP32 values as bias_f32 followed by the consumer, and a
// producer that splits its output with cvt.rna computes the same high/low
// parts as split_tf32 after the producer.

struct h3_gpu_event {
    cudaEvent_t event{};
};

extern "C" h3_gpu_event *h3_gpu_event_new(h3_gpu *g) {
    auto *e = new h3_gpu_event;
    // Blocking sync lets a waiting host thread sleep instead of spinning.
    if (!cuda_ok(g, cudaEventCreateWithFlags(&e->event,
                                             cudaEventDisableTiming | cudaEventBlockingSync))) {
        delete e;
        return nullptr;
    }
    return e;
}
extern "C" void h3_gpu_event_free(h3_gpu_event *e) {
    if (!e) return;
    cudaEventDestroy(e->event);
    delete e;
}
// Safe to call from another host thread: it touches no h3_gpu state.
extern "C" int h3_gpu_event_wait(h3_gpu_event *e, char *error, size_t size) {
    cudaError_t r = e ? cudaEventSynchronize(e->event) : cudaErrorInvalidValue;
    if (r == cudaSuccess) return 1;
    if (error && size) snprintf(error, size, "CUDA: %s", cudaGetErrorString(r));
    return 0;
}
extern "C" void *h3_gpu_host_alloc(size_t bytes) {
    void *p = nullptr;
    return cudaHostAlloc(&p, std::max(bytes, size_t(1)), cudaHostAllocDefault) == cudaSuccess
        ? p : nullptr;
}
extern "C" void h3_gpu_host_free(void *p) {
    if (p) cudaFreeHost(p);
}
// Host-to-device copy ordered on the compute stream, without a host wait.
// The host buffer must stay unchanged until the stream has passed the copy.
extern "C" int h3_gpu_tensor_upload_f32_async(h3_gpu *g, h3_gpu_tensor *t, const float *host,
                                              size_t n) {
    if (!h3_gpu_require_elements(g, t, n, "async upload") || t->dtype != H3_GPU_F32) return 0;
    return cuda_ok(g, cudaMemcpyAsync(t->data, host, n * sizeof(float),
                                      cudaMemcpyHostToDevice, g->stream));
}
// Device-to-host copy of the work queued so far, on a separate copy stream so
// it overlaps later compute. `done` fires when the host buffer is filled. The
// caller must not let the compute stream overwrite the tensor before that:
// see h3_gpu_stream_wait_event.
extern "C" int h3_gpu_tensor_download_f32_async(h3_gpu *g, const h3_gpu_tensor *t, float *host,
                                                size_t n, h3_gpu_event *done) {
    if (!h3_gpu_require_elements(g, t, n, "async download") || t->dtype != H3_GPU_F32 || !done)
        return 0;
    if (!g->copy_stream &&
        !cuda_ok(g, cudaStreamCreateWithFlags(&g->copy_stream, cudaStreamNonBlocking)))
        return 0;
    if (!g->copy_ready &&
        !cuda_ok(g, cudaEventCreateWithFlags(&g->copy_ready, cudaEventDisableTiming)))
        return 0;
    return cuda_ok(g, cudaEventRecord(g->copy_ready, g->stream)) &&
           cuda_ok(g, cudaStreamWaitEvent(g->copy_stream, g->copy_ready, 0)) &&
           cuda_ok(g, cudaMemcpyAsync(host, t->data, n * sizeof(float), cudaMemcpyDeviceToHost,
                                      g->copy_stream)) &&
           cuda_ok(g, cudaEventRecord(done->event, g->copy_stream));
}
// Later compute-stream work waits for the event (no host wait).
extern "C" int h3_gpu_stream_wait_event(h3_gpu *g, h3_gpu_event *e) {
    return e && cuda_ok(g, cudaStreamWaitEvent(g->stream, e->event, 0));
}
extern "C" int h3_gpu_tensor_is_split_tf32(const h3_gpu_tensor *t) {
    return t && t->tf32_low;
}

__device__ inline void tf32_parts(float v, float *high, float *low, size_t i) {
    uint bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(v));
    float h = __uint_as_float(bits);
    high[i] = h;
    low[i] = v - h;
}
// h3_rms_norm_f32 with the output written as TF32 high/low parts.
__global__ void rms_norm_split_tf32(const float *input, const float *weight, float *high,
                                    float *low, norm_args args) {
    uint row = blockIdx.x, tid = threadIdx.x, threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads)
        local_sum = fmaf(x[k], x[k], local_sum);
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        tf32_parts(x[column] * inverse * weight[column], high, low,
                   size_t(row) * args.width + column);
}
// h3_video_qkv_rope_f32 on qkv + bias.
__global__ void video_qkv_rope_bias(const float *qkv, const float *bias, const float *rope_cos,
                                    const float *rope_sin, float *query, float *key,
                                    float *value, qkv_args args) {
    uint dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint head = blockIdx.y * blockDim.y + threadIdx.y;
    uint row = blockIdx.z * blockDim.z + threadIdx.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint column = head * args.head_dim * 3;
    uint base = row * args.heads * args.head_dim * 3 + column;
    const float *b = bias + column;
#define QKV(offset) __fadd_rn(qkv[base + (offset)], b[offset])
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = QKV(d);
        float k = QKV(args.head_dim + d);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float qi = rsqrtf(q_sum / float(args.head_dim) + args.epsilon);
    float ki = rsqrtf(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = QKV(dimension) * qi;
    float k0 = QKV(args.head_dim + dimension) * ki;
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = QKV(pair) * qi;
        float k1 = QKV(args.head_dim + pair) * ki;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = QKV(pair) * qi;
        float k1 = QKV(args.head_dim + pair) * ki;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = QKV(args.head_dim * 2 + dimension);
#undef QKV
}
// The same values with one block per row and `heads_per_block` heads: each
// element is loaded once into shared memory, and one thread per head runs
// each sum of squares in the original sequential order (h3_video_qkv_rope_f32
// makes every thread repeat both 64-term sums from global memory).
__global__ void video_qkv_rope_bias_rows(const float *qkv, const float *bias,
                                         const float *rope_cos, const float *rope_sin,
                                         float *query, float *key, float *value,
                                         qkv_args args, uint heads_per_block) {
    extern __shared__ float shared[];
    uint width = args.head_dim * 3;
    float *inverses = shared + heads_per_block * width;
    uint row = blockIdx.x, head0 = blockIdx.y * heads_per_block, tid = threadIdx.x;
    size_t row_base = size_t(row) * args.heads * width + size_t(head0) * width;
    for (uint i = tid; i < heads_per_block * width; i += blockDim.x)
        shared[i] = __fadd_rn(qkv[row_base + i], bias[head0 * width + i]);
    __syncthreads();
    uint local = tid / args.head_dim, dimension = tid % args.head_dim;
    const float *x = shared + local * width;
    if (dimension < 2) {
        const float *part = x + dimension * args.head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) sum = fmaf(part[d], part[d], sum);
        inverses[local * 2 + dimension] = rsqrtf(sum / float(args.head_dim) + args.epsilon);
    }
    __syncthreads();
    float qi = inverses[local * 2], ki = inverses[local * 2 + 1];
    float q0 = x[dimension] * qi;
    float k0 = x[args.head_dim + dimension] * ki;
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = x[pair] * qi;
        float k1 = x[args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = x[pair] * qi;
        float k1 = x[args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head0 + local) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = x[args.head_dim * 2 + dimension];
}
// h3_swiglu_f32 on fused + bias, output written as TF32 high/low parts.
__global__ void swiglu_bias_split_tf32(const float *fused, const float *bias, float *high,
                                       float *low, swiglu_args args) {
    uint column = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = __fadd_rn(fused[base + column], bias[column]);
    float up = __fadd_rn(fused[base + args.width + column], bias[args.width + column]);
    tf32_parts(gate / (1.0f + expf(-gate)) * up, high, low, size_t(row) * args.width + column);
}
// h3_scale_add_f32 on branch + bias.
__global__ void scale_add_bias(const float *residual, const float *branch, const float *bias,
                               const float *scale, float *output, scale_add_args args) {
    uint column = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint index = row * args.width + column;
    output[index] = residual[index] + __fadd_rn(branch[index], bias[column]) * scale[column];
}

static bool f32_tensor(h3_gpu *g, const h3_gpu_tensor *t, size_t n, const char *what) {
    if (!h3_gpu_require_elements(g, t, n, what)) return false;
    if (t->dtype == H3_GPU_F32) return true;
    h3_gpu_set_error(g, "expected FP32: %s", what);
    return false;
}
extern "C" int h3_gpu_rms_norm_split_tf32(h3_gpu *g, h3_gpu_tensor *parts,
                                          const h3_gpu_tensor *x, const h3_gpu_tensor *w,
                                          uint rows, uint width, float eps) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, x, count, "RMSNorm input") || !f32_tensor(g, w, width, "RMSNorm weight") ||
        !f32_tensor(g, parts, 2 * count, "RMSNorm TF32 parts")) return 0;
    float *high = (float *)parts->data;
    rms_norm_split_tf32<<<rows, 256, 0, g->stream>>>((const float *)x->data,
        (const float *)w->data, high, high + count, norm_args{rows, width, eps});
    return launch_ok(g);
}
extern "C" int h3_gpu_swiglu_bias_split_tf32(h3_gpu *g, h3_gpu_tensor *parts,
                                             const h3_gpu_tensor *fused,
                                             const h3_gpu_tensor *bias, uint rows,
                                             uint width) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, fused, 2 * count, "SwiGLU input") ||
        !f32_tensor(g, bias, 2 * size_t(width), "SwiGLU bias") ||
        !f32_tensor(g, parts, 2 * count, "SwiGLU TF32 parts")) return 0;
    float *high = (float *)parts->data;
    swiglu_bias_split_tf32<<<dim3((width + 31) / 32, (rows + 7) / 8), dim3(32, 8), 0,
                             g->stream>>>((const float *)fused->data, (const float *)bias->data,
                                          high, high + count, swiglu_args{rows, width});
    return launch_ok(g);
}
extern "C" int h3_gpu_scale_add_bias_f32(h3_gpu *g, h3_gpu_tensor *output,
                                         const h3_gpu_tensor *residual,
                                         const h3_gpu_tensor *branch,
                                         const h3_gpu_tensor *bias,
                                         const h3_gpu_tensor *scale, uint rows, uint width) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, residual, count, "scale-add residual") ||
        !f32_tensor(g, branch, count, "scale-add branch") ||
        !f32_tensor(g, bias, width, "scale-add bias") ||
        !f32_tensor(g, scale, width, "scale-add scale") ||
        !f32_tensor(g, output, count, "scale-add output")) return 0;
    scale_add_bias<<<dim3((width + 31) / 32, (rows + 7) / 8), dim3(32, 8), 0, g->stream>>>(
        (const float *)residual->data, (const float *)branch->data, (const float *)bias->data,
        (const float *)scale->data, (float *)output->data, scale_add_args{rows, width});
    return launch_ok(g);
}
extern "C" int h3_gpu_video_qkv_rope_bias_f32(h3_gpu *g, h3_gpu_tensor *query,
                                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                                              const h3_gpu_tensor *qkv,
                                              const h3_gpu_tensor *bias,
                                              const h3_gpu_tensor *rope_cos,
                                              const h3_gpu_tensor *rope_sin, uint sequence,
                                              uint heads, uint head_dim, uint rope_half,
                                              float eps) {
    size_t inner = size_t(heads) * head_dim, count = size_t(sequence) * inner;
    size_t rope_count = size_t(sequence) * rope_half;
    if (!f32_tensor(g, qkv, 3 * count, "video QKV") ||
        !f32_tensor(g, bias, 3 * inner, "video QKV bias") ||
        !f32_tensor(g, rope_cos, rope_count, "video RoPE cosine") ||
        !f32_tensor(g, rope_sin, rope_count, "video RoPE sine") ||
        !f32_tensor(g, query, count, "video query") || !f32_tensor(g, key, count, "video key") ||
        !f32_tensor(g, value, count, "video value") || rope_half * 2 > head_dim) return 0;
    qkv_args args{sequence, heads, head_dim, rope_half, 0, eps};
    if (head_dim >= 2 && head_dim <= 256 && 256 % head_dim == 0 && heads % (256 / head_dim) == 0) {
        uint per_block = 256 / head_dim;
        size_t shared = (size_t(per_block) * head_dim * 3 + per_block * 2) * sizeof(float);
        video_qkv_rope_bias_rows<<<dim3(sequence, heads / per_block), 256, shared, g->stream>>>(
            (const float *)qkv->data, (const float *)bias->data, (const float *)rope_cos->data,
            (const float *)rope_sin->data, (float *)query->data, (float *)key->data,
            (float *)value->data, args, per_block);
        return launch_ok(g);
    }
    // Same block shape rule as h3_gpu_dispatch_3d.
    unsigned by = 4, bz = 1;
    while ((heads + by - 1) / by > 65535 && 32 * by * bz < 1024) by *= 2;
    while ((sequence + bz - 1) / bz > 65535 && 32 * by * bz < 1024) bz *= 2;
    if ((heads + by - 1) / by > 65535 || (sequence + bz - 1) / bz > 65535) {
        h3_gpu_set_error(g, "video QKV bias: grid too large");
        return 0;
    }
    video_qkv_rope_bias<<<dim3((head_dim + 31) / 32, (heads + by - 1) / by,
                               (sequence + bz - 1) / bz), dim3(32, by, bz), 0, g->stream>>>(
        (const float *)qkv->data, (const float *)bias->data, (const float *)rope_cos->data,
        (const float *)rope_sin->data, (float *)query->data, (float *)key->data,
        (float *)value->data, qkv_args{sequence, heads, head_dim, rope_half, 0, eps});
    return launch_ok(g);
}
// y = x w^T with x given as TF32 parts (high then low, 2*rows*k elements)
// and w split at load. No bias: the consumer adds it.
extern "C" int h3_gpu_linear_split_input_f32(h3_gpu *g, h3_gpu_tensor *y,
                                             const h3_gpu_tensor *parts,
                                             const h3_gpu_tensor *w, uint rows, uint k,
                                             uint n) {
    size_t count = size_t(rows) * k;
    if (!rows || !k || !n || rows > INT_MAX || k > INT_MAX || n > INT_MAX ||
        !f32_tensor(g, parts, 2 * count, "split linear input") ||
        !f32_tensor(g, w, size_t(n) * k, "split linear weight") ||
        !f32_tensor(g, y, size_t(rows) * n, "split linear output")) return 0;
    if (!w->tf32_low) {
        h3_gpu_set_error(g, "split linear needs a 3xTF32 weight");
        return 0;
    }
    const float *high = (const float *)parts->data;
    return gemm_3xtf32(g, y, high, high + count, w, rows, k, n);
}
