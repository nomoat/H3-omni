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
// The same for head dimension 64 with an even RoPE half: one warp per
// (row, head), each lane owning two adjacent dimensions. The QK-norm sums are
// warp reductions instead of one thread's serial sum; the RoPE partner of
// dimension d (d +/- rope_half) lives on lane +/- rope_half / 2.
__global__ void video_qkv_rope_bias_f16_d64(const float *qkv, const float *bias,
                                            const float *rope_cos, const float *rope_sin,
                                            __half *query, __half *key, __half *value,
                                            qkv_args args) {
    uint lane = threadIdx.x & 31;
    size_t item = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (item >= size_t(args.sequence) * args.heads) return;
    uint row = uint(item / args.heads), head = uint(item % args.heads);
    const float *x = qkv + item * 192;
    const float *b = bias + size_t(head) * 192;
    uint d = 2 * lane;
    float2 q = *reinterpret_cast<const float2 *>(x + d);
    float2 k = *reinterpret_cast<const float2 *>(x + 64 + d);
    float2 v = *reinterpret_cast<const float2 *>(x + 128 + d);
    q.x = __fadd_rn(q.x, b[d]); q.y = __fadd_rn(q.y, b[d + 1]);
    k.x = __fadd_rn(k.x, b[64 + d]); k.y = __fadd_rn(k.y, b[64 + d + 1]);
    v.x = __fadd_rn(v.x, b[128 + d]); v.y = __fadd_rn(v.y, b[128 + d + 1]);
    float qs = fmaf(q.y, q.y, q.x * q.x), ks = fmaf(k.y, k.y, k.x * k.x);
    for (int offset = 16; offset; offset >>= 1) {
        qs += __shfl_xor_sync(0xffffffffu, qs, offset);
        ks += __shfl_xor_sync(0xffffffffu, ks, offset);
    }
    float qi = rsqrtf(qs / 64.0f + args.epsilon), ki = rsqrtf(ks / 64.0f + args.epsilon);
    uint shift = args.rope_half / 2;
    uint partner = lane < shift ? lane + shift : lane < 2 * shift ? lane - shift : lane;
    float2 qp = make_float2(__shfl_sync(0xffffffffu, q.x, partner),
                            __shfl_sync(0xffffffffu, q.y, partner));
    float2 kp = make_float2(__shfl_sync(0xffffffffu, k.x, partner),
                            __shfl_sync(0xffffffffu, k.y, partner));
    float q0 = q.x * qi, q1 = q.y * qi, k0 = k.x * ki, k1 = k.y * ki;
    if (d < 2 * args.rope_half) {
        float sign = d < args.rope_half ? -1.0f : 1.0f;
        uint r = d < args.rope_half ? d : d - args.rope_half;
        const float *c = rope_cos + size_t(row) * args.rope_half + r;
        const float *s = rope_sin + size_t(row) * args.rope_half + r;
        float pq0 = qp.x * qi, pq1 = qp.y * qi, pk0 = kp.x * ki, pk1 = kp.y * ki;
        if (sign < 0) {
            q0 = q0 * c[0] - pq0 * s[0]; q1 = q1 * c[1] - pq1 * s[1];
            k0 = k0 * c[0] - pk0 * s[0]; k1 = k1 * c[1] - pk1 * s[1];
        } else {
            q0 = q0 * c[0] + pq0 * s[0]; q1 = q1 * c[1] + pq1 * s[1];
            k0 = k0 * c[0] + pk0 * s[0]; k1 = k1 * c[1] + pk1 * s[1];
        }
    }
    size_t out = item * 64 + d;
    *reinterpret_cast<__half2 *>(query + out) = __floats2half2_rn(q0, q1);
    *reinterpret_cast<__half2 *>(key + out) = __floats2half2_rn(k0, k1);
    *reinterpret_cast<__half2 *>(value + out) = __floats2half2_rn(v.x, v.y);
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
    if (head_dim == 64 && rope_half % 2 == 0) {
        size_t items = size_t(sequence) * heads;
        video_qkv_rope_bias_f16_d64<<<(items + 7) / 8, 256, 0, g->stream>>>(
            (const float *)qkv->data, (const float *)bias->data, (const float *)rope_cos->data,
            (const float *)rope_sin->data, (__half *)query->data, (__half *)key->data,
            (__half *)value->data, qkv_args{sequence, heads, head_dim, rope_half, 0, eps});
        return launch_ok(g);
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
// The same with four columns per thread (width divisible by 4).
__global__ void swiglu_bias_f16_out4(const float *fused, const float *bias, __half *output,
                                     swiglu_args args) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint quads = args.width / 4;
    if (index >= size_t(args.rows) * quads) return;
    uint row = uint(index / quads), column = uint(index % quads) * 4;
    size_t base = size_t(row) * args.width * 2;
    float4 g = *reinterpret_cast<const float4 *>(fused + base + column);
    float4 u = *reinterpret_cast<const float4 *>(fused + base + args.width + column);
    float4 gb = *reinterpret_cast<const float4 *>(bias + column);
    float4 ub = *reinterpret_cast<const float4 *>(bias + args.width + column);
    float g0 = __fadd_rn(g.x, gb.x), g1 = __fadd_rn(g.y, gb.y);
    float g2 = __fadd_rn(g.z, gb.z), g3 = __fadd_rn(g.w, gb.w);
    float u0 = __fadd_rn(u.x, ub.x), u1 = __fadd_rn(u.y, ub.y);
    float u2 = __fadd_rn(u.z, ub.z), u3 = __fadd_rn(u.w, ub.w);
    __half2 *out = reinterpret_cast<__half2 *>(output + size_t(row) * args.width + column);
    out[0] = __floats2half2_rn(g0 / (1.0f + expf(-g0)) * u0, g1 / (1.0f + expf(-g1)) * u1);
    out[1] = __floats2half2_rn(g2 / (1.0f + expf(-g2)) * u2, g3 / (1.0f + expf(-g3)) * u3);
}
extern "C" int h3_gpu_swiglu_bias_f16_out(h3_gpu *g, h3_gpu_tensor *output,
                                          const h3_gpu_tensor *fused, const h3_gpu_tensor *bias,
                                          uint rows, uint width) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, fused, 2 * count, "SwiGLU input") ||
        !f32_tensor(g, bias, 2 * size_t(width), "SwiGLU bias") ||
        !f16_tensor(g, output, count, "SwiGLU FP16 output")) return 0;
    if (width % 4 == 0) {
        size_t quads = count / 4;
        swiglu_bias_f16_out4<<<(quads + 255) / 256, 256, 0, g->stream>>>(
            (const float *)fused->data, (const float *)bias->data, (__half *)output->data,
            swiglu_args{rows, width});
        return launch_ok(g);
    }
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

// scale_add_bias followed by rms_norm_f16_out in one pass: output =
// residual + (branch + bias) * scale (FP32, may alias residual), then the
// FP16 RMSNorm of it. Same per-element arithmetic and reduction order as the
// two kernels, so the values are identical; the residual row is read once.
__global__ void scale_add_bias_rms_f16(const float *residual, const float *branch,
                                       const float *bias, const float *scale, float *output,
                                       const float *weight, __half *normed, norm_args args) {
    uint row = blockIdx.x, tid = threadIdx.x, threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    size_t base = size_t(row) * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float x = residual[base + k] + __fadd_rn(branch[base + k], bias[k]) * scale[k];
        output[base + k] = x;
        local_sum = fmaf(x, x, local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        normed[base + column] = __float2half_rn(output[base + column] * inverse * weight[column]);
}
extern "C" int h3_gpu_scale_add_bias_rms_f16(h3_gpu *g, h3_gpu_tensor *output,
                                             h3_gpu_tensor *normed,
                                             const h3_gpu_tensor *residual,
                                             const h3_gpu_tensor *branch,
                                             const h3_gpu_tensor *bias,
                                             const h3_gpu_tensor *scale,
                                             const h3_gpu_tensor *weight, uint rows,
                                             uint width, float eps) {
    size_t count = size_t(rows) * width;
    if (!f32_tensor(g, residual, count, "scale-add residual") ||
        !f32_tensor(g, branch, count, "scale-add branch") ||
        !f32_tensor(g, bias, width, "scale-add bias") ||
        !f32_tensor(g, scale, width, "scale-add scale") ||
        !f32_tensor(g, output, count, "scale-add output") ||
        !f32_tensor(g, weight, width, "RMSNorm weight") ||
        !f16_tensor(g, normed, count, "RMSNorm FP16 output")) return 0;
    scale_add_bias_rms_f16<<<rows, 256, 0, g->stream>>>((const float *)residual->data,
        (const float *)branch->data, (const float *)bias->data, (const float *)scale->data,
        (float *)output->data, (const float *)weight->data, (__half *)normed->data,
        norm_args{rows, width, eps});
    return launch_ok(g);
}
