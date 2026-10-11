// Video VAE FP16 mode: the four block projections and the attention run on
// FP16 tensor cores with FP32 accumulation, as in the diffusers mixed-precision
// decode. Norm statistics, QK norm, RoPE, SwiGLU, the biases and the residual
// stream stay FP32; only the inputs of the products are rounded to FP16, and
// the products return FP32. Included at the end of h3_gpu_cuda.cu.

static bool f16_tensor(h3_gpu *g, const h3_gpu_tensor *t, size_t n, const char *what) {
    if (!h3_gpu_require_elements(g, t, n, what)) return false;
    if (t->dtype == H3_GPU_F16) return true;
    h3_gpu_set_error(g, "expected FP16: %s", what);
    return false;
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_f16(h3_gpu *g, size_t n) {
    return allocate(g, n, H3_GPU_F16);
}

__global__ void f32_to_f16(const float *x, __half *y, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
// A new FP16 tensor rounded from an FP32 one (weights, at load).
extern "C" h3_gpu_tensor *h3_gpu_tensor_to_f16(h3_gpu *g, const h3_gpu_tensor *x) {
    if (!f32_tensor(g, x, x ? x->elements : 0, "FP16 conversion input")) return nullptr;
    auto *y = allocate(g, x->elements, H3_GPU_F16);
    if (!y) return nullptr;
    f32_to_f16<<<(x->elements + 255) / 256, 256, 0, g->stream>>>(
        (const float *)x->data, (__half *)y->data, x->elements);
    if (!launch_ok(g)) {
        h3_gpu_tensor_free(y);
        return nullptr;
    }
    return y;
}

// h3_rms_norm_f32 with the output rounded to FP16.
__global__ void rms_norm_f16_out(const float *input, const float *weight, __half *output,
                                 norm_args args) {
    uint row = blockIdx.x, tid = threadIdx.x, threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const float *x = input + size_t(row) * args.width;
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
        output[size_t(row) * args.width + column] =
            __float2half_rn(x[column] * inverse * weight[column]);
}
extern "C" int h3_gpu_rms_norm_f16_out(h3_gpu *g, h3_gpu_tensor *output, const h3_gpu_tensor *x,
                                       const h3_gpu_tensor *w, uint rows, uint width,
                                       float eps) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, x, count, "RMSNorm input") || !f32_tensor(g, w, width, "RMSNorm weight") ||
        !f16_tensor(g, output, count, "RMSNorm FP16 output")) return 0;
    rms_norm_f16_out<<<rows, 256, 0, g->stream>>>((const float *)x->data,
        (const float *)w->data, (__half *)output->data, norm_args{rows, width, eps});
    return launch_ok(g);
}

// video_qkv_rope_bias_rows with FP16 query, key and value.
__global__ void video_qkv_rope_bias_f16(const float *qkv, const float *bias,
                                        const float *rope_cos, const float *rope_sin,
                                        __half *query, __half *key, __half *value,
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
    size_t output_index = (size_t(row) * args.heads + head0 + local) * args.head_dim + dimension;
    query[output_index] = __float2half_rn(q0);
    key[output_index] = __float2half_rn(k0);
    value[output_index] = __float2half_rn(x[args.head_dim * 2 + dimension]);
}
extern "C" int h3_gpu_video_qkv_rope_bias_f16(h3_gpu *g, h3_gpu_tensor *query,
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
        !f16_tensor(g, query, count, "video FP16 query") ||
        !f16_tensor(g, key, count, "video FP16 key") ||
        !f16_tensor(g, value, count, "video FP16 value")) return 0;
    if (rope_half * 2 > head_dim || head_dim < 2 || head_dim > 256 || 256 % head_dim ||
        heads % (256 / head_dim)) {
        h3_gpu_set_error(g, "video FP16 QKV: unsupported head shape");
        return 0;
    }
    uint per_block = 256 / head_dim;
    size_t shared = (size_t(per_block) * head_dim * 3 + per_block * 2) * sizeof(float);
    video_qkv_rope_bias_f16<<<dim3(sequence, heads / per_block), 256, shared, g->stream>>>(
        (const float *)qkv->data, (const float *)bias->data, (const float *)rope_cos->data,
        (const float *)rope_sin->data, (__half *)query->data, (__half *)key->data,
        (__half *)value->data, qkv_args{sequence, heads, head_dim, rope_half, 0, eps}, per_block);
    return launch_ok(g);
}

// h3_swiglu_f32 on fused + bias, output rounded to FP16.
__global__ void swiglu_bias_f16_out(const float *fused, const float *bias, __half *output,
                                    swiglu_args args) {
    uint column = blockIdx.x * blockDim.x + threadIdx.x;
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    size_t base = size_t(row) * args.width * 2;
    float gate = __fadd_rn(fused[base + column], bias[column]);
    float up = __fadd_rn(fused[base + args.width + column], bias[args.width + column]);
    output[size_t(row) * args.width + column] = __float2half_rn(gate / (1.0f + expf(-gate)) * up);
}
extern "C" int h3_gpu_swiglu_bias_f16_out(h3_gpu *g, h3_gpu_tensor *output,
                                          const h3_gpu_tensor *fused, const h3_gpu_tensor *bias,
                                          uint rows, uint width) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, fused, 2 * count, "SwiGLU input") ||
        !f32_tensor(g, bias, 2 * size_t(width), "SwiGLU bias") ||
        !f16_tensor(g, output, count, "SwiGLU FP16 output")) return 0;
    swiglu_bias_f16_out<<<dim3((width + 31) / 32, (rows + 7) / 8), dim3(32, 8), 0, g->stream>>>(
        (const float *)fused->data, (const float *)bias->data, (__half *)output->data,
        swiglu_args{rows, width});
    return launch_ok(g);
}

// y (FP32) = x (FP16) w^T (FP16), FP32 accumulation. No bias: the consumer
// adds it in FP32.
extern "C" int h3_gpu_linear_f16_f32_out(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                                         const h3_gpu_tensor *w, uint rows, uint k, uint n) {
    if (!rows || !k || !n || rows > INT_MAX || k > INT_MAX || n > INT_MAX ||
        !f16_tensor(g, x, size_t(rows) * k, "FP16 linear input") ||
        !f16_tensor(g, w, size_t(n) * k, "FP16 linear weight") ||
        !f32_tensor(g, y, size_t(rows) * n, "FP16 linear output")) return 0;
    float one = 1, zero = 0;
    ++g->stats.mps_linear_dispatches;
    return blas_ok(g, cublasGemmEx(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, w->data,
                                   CUDA_R_16F, k, x->data, CUDA_R_16F, k, &zero, y->data,
                                   CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Non-causal FP16 attention, head dimension 64, token-major
// [sequence, heads, 64] inputs and output (FlashAttention-3).
extern "C" int h3_gpu_sdpa_f16_d64(h3_gpu *g, h3_gpu_tensor *output, const h3_gpu_tensor *q,
                                   const h3_gpu_tensor *k, const h3_gpu_tensor *v, uint sequence,
                                   uint heads, uint head_dim, float scale) {
    size_t count = size_t(sequence) * heads * head_dim;
    if (head_dim != 64 || !sequence || !heads) {
        h3_gpu_set_error(g, "FP16 attention supports head dimension 64 only");
        return 0;
    }
    if (!f16_tensor(g, q, count, "FP16 attention query") ||
        !f16_tensor(g, k, count, "FP16 attention key") ||
        !f16_tensor(g, v, count, "FP16 attention value") ||
        !f16_tensor(g, output, count, "FP16 attention output")) return 0;
    auto *lse = allocate(g, size_t(heads) * sequence, H3_GPU_F32);
    if (!lse) return 0;
    int ok = cuda_ok(g, h3_flash3_f16_d64(output->data, q->data, k->data, v->data,
                                          static_cast<float *>(lse->data), 1, sequence, heads,
                                          scale, g->stream));
    h3_gpu_tensor_free(lse);
    ++g->stats.mps_sdpa_dispatches;
    return ok;
}
