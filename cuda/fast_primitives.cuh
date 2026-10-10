// Hand-written (NOT generated) fast versions of the DiT elementwise kernels.
//
// cuda/translate_primitives.py generates the scalar reference kernels in
// primitives.cuh; regeneration does not touch this file. Every kernel here
// performs the same per-element float operations, in the same order and with
// the same BF16 rounding points, as its generated counterpart (the generated
// code with the rounding edits recorded in the generator). Only the thread
// mapping changes: vector loads/stores, one warp per attention head, and the
// gate fused into the following AdaLN. The build uses --fmad=false, so the
// compiler cannot contract a multiply and an add differently in either version.
//
// Reductions keep the generated order exactly:
//  - RMS over a DiT row (rms_sum_bf16): thread t < 128 owns the 4-value groups
//    starting at 4t + 512j, sums them with fmaf in index order, then the same
//    shfl_down tree, the same warp_sums pairing (0+2, 1+3, then 0+1).
//  - Per-head q/k RMS (head_dim 128): lane t sums d = 4t..4t+3 with fmaf in
//    index order, then the same shfl_down tree and a broadcast from lane 0.
// If you change a generated kernel's arithmetic in the generator, change the
// matching kernel here too; tests/serving/profile/elementwise_bench.cu checks
// every fast kernel bit for bit against the generated one.

struct fast_gate_adaln_args {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

__device__ inline void h3_unpack4(uint2 v, float out[4]) {
    out[0] = h3_bf16_to_f32((ushort)(v.x & 0xffffu));
    out[1] = h3_bf16_to_f32((ushort)(v.x >> 16));
    out[2] = h3_bf16_to_f32((ushort)(v.y & 0xffffu));
    out[3] = h3_bf16_to_f32((ushort)(v.y >> 16));
}
__device__ inline uint2 h3_pack4(const ushort b[4]) {
    return make_uint2(uint(b[0]) | (uint(b[1]) << 16), uint(b[2]) | (uint(b[3]) << 16));
}

// Same arithmetic as h3_gate_bf16 followed by h3_adaln_bf16 on the gate output
// (GATE = true), or as h3_adaln_bf16 alone (GATE = false). One 128-thread block
// per row; requires width % 4 == 0, width <= 512 * CHUNKS and 8-byte aligned
// pointers. The gated row stays in registers, so the AdaLN does not read it back.
// `residual_out` may alias `residual`, and `output` may alias `input`: every
// thread reads its own values before it writes the same positions.
template <int CHUNKS, bool GATE>
__global__ void __launch_bounds__(128)
h3_fast_gate_adaln_bf16(const ushort *input, const ushort *branch, const ushort *gate_modulation,
                        ushort *residual_out, const ushort *weight, const ushort *modulation,
                        const uint *row_map, ushort *output, fast_gate_adaln_args args) {
    const uint row = blockIdx.x;
    const uint tid = threadIdx.x;
    if (row >= args.rows) return;
    __shared__ float warp_sums[4];
    const uint width = args.width;
    const uint mod_row = row_map[row];
    const ushort *x = input + row * width;
    uint2 held[CHUNKS];
    float sum = 0;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const uint start = 4 * tid + 512 * c;
        if (start < width) {
            uint2 raw = *(const uint2 *)(x + start);
            if (GATE) {
                // h3_gate_bf16: residual + bf16(branch * gate), rounded to BF16.
                float r[4], b[4], g[4];
                h3_unpack4(raw, r);
                h3_unpack4(*(const uint2 *)(branch + row * width + start), b);
                h3_unpack4(*(const uint2 *)(gate_modulation + mod_row * args.slots * width +
                                            args.gate_slot * width + start), g);
                ushort o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float value = r[i] + bf16_round(b[i] * g[i]);
                    o[i] = h3_f32_to_bf16(value);
                }
                raw = h3_pack4(o);
                *(uint2 *)(residual_out + row * width + start) = raw;
            }
            held[c] = raw;
            float v[4];
            h3_unpack4(raw, v);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                sum = fmaf(v[i], v[i], sum);
        }
    }
    // rms_sum_bf16's reduction, unchanged.
    for (uint distance = 16; distance; distance >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, distance);
    if (tid % 32 == 0) warp_sums[tid / 32] = sum;
    __syncthreads();
    if (tid < 2) warp_sums[tid] += warp_sums[tid + 2];
    __syncthreads();
    if (tid == 0) warp_sums[0] += warp_sums[1];
    __syncthreads();
    const float total = warp_sums[0];
    const float inverse = rsqrtf(total / float(width) + args.epsilon);
    const uint base = mod_row * args.slots * width;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const uint start = 4 * tid + 512 * c;
        if (start < width) {
            float xv[4], w[4], sh[4], sc[4];
            h3_unpack4(held[c], xv);
            h3_unpack4(*(const uint2 *)(weight + start), w);
            h3_unpack4(*(const uint2 *)(modulation + base + args.shift_slot * width + start), sh);
            h3_unpack4(*(const uint2 *)(modulation + base + args.scale_slot * width + start), sc);
            ushort o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float normalized = xv[i] * inverse * w[i];
                float shift = sh[i];
                float scale = sc[i];
                o[i] = h3_f32_to_bf16(
                    bf16_round(bf16_round(normalized) * bf16_round(1.0f + scale)) + shift);
            }
            *(uint2 *)(output + row * width + start) = h3_pack4(o);
        }
    }
}

// Same arithmetic as h3_gate_bf16, eight values per thread (16-byte accesses).
// Requires width % 8 == 0 and 16-byte aligned pointers.
__global__ void __launch_bounds__(256)
h3_fast_gate_bf16(const ushort *residual, const ushort *branch, const ushort *modulation,
                  const uint *row_map, ushort *output, gate_args args) {
    const size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t first = group * 8;
    if (first >= (size_t)args.rows * args.width) return;
    const uint row = (uint)(first / args.width);
    const uint column = (uint)(first - (size_t)row * args.width);
    const uint base = row_map[row] * args.slots * args.width;
    const uint index = row * args.width + column;
    uint4 rr = *(const uint4 *)(residual + index);
    uint4 bb = *(const uint4 *)(branch + index);
    uint4 gg = *(const uint4 *)(modulation + base + args.gate_slot * args.width + column);
    const ushort *r = (const ushort *)&rr, *b = (const ushort *)&bb, *g = (const ushort *)&gg;
    uint4 out;
    ushort *o = (ushort *)&out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float gate = h3_bf16_to_f32(g[i]);
        float value = h3_bf16_to_f32(r[i]) + bf16_round(h3_bf16_to_f32(b[i]) * gate);
        o[i] = h3_f32_to_bf16(value);
    }
    *(uint4 *)(output + index) = out;
}

// Same arithmetic as h3_swiglu_bf16, eight values per thread (16-byte accesses).
// Requires width % 8 == 0 and 16-byte aligned pointers.
__global__ void __launch_bounds__(256)
h3_fast_swiglu_bf16(const ushort *fused, ushort *output, swiglu_args args) {
    const size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t first = group * 8;
    if (first >= (size_t)args.rows * args.width) return;
    const uint row = (uint)(first / args.width);
    const uint column = (uint)(first - (size_t)row * args.width);
    const uint base = row * args.width * 2;
    uint4 gg = *(const uint4 *)(fused + base + column);
    uint4 uu = *(const uint4 *)(fused + base + args.width + column);
    const ushort *g = (const ushort *)&gg, *u = (const ushort *)&uu;
    uint4 out;
    ushort *o = (ushort *)&out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float gate = h3_bf16_to_f32(g[i]);
        float up = h3_bf16_to_f32(u[i]);
        o[i] = h3_f32_to_bf16(bf16_round(gate / (1.0f + expf(-gate))) * up);
    }
    *(uint4 *)(output + row * args.width + column) = out;
}


// Same arithmetic as h3_qkv_rope_bf16 for head_dim == 128: one warp per
// (row, head), lane t owns dimensions 4t..4t+3 (the generated reduction's
// grouping). Requires rope_half % 4 == 0, 2 * rope_half <= 128 and 8-byte
// aligned pointers. The generated kernel recomputes the RoPE partner value
// bf16(x[pair] * inverse * weight[pair]); that is exactly the partner lane's
// own rounded value, so it is taken with a warp shuffle instead.
__global__ void __launch_bounds__(256)
h3_fast_qkv_rope_bf16(const ushort *qkv, const ushort *q_weight, const ushort *k_weight,
                      const ushort *rope_cos, const ushort *rope_sin, ushort *query,
                      ushort *key, ushort *value, qkv_args args) {
    const uint lane = threadIdx.x & 31;
    const size_t item = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (item >= (size_t)args.sequence * args.heads) return;  // whole warp leaves together
    const uint row = (uint)(item / args.heads);
    const uint head = (uint)(item - (size_t)row * args.heads);
    const uint inner = args.heads * args.head_dim;
    const uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    const uint d0 = lane * 4;
    float qx[4], kx[4];
    h3_unpack4(*(const uint2 *)(qkv + q_base + d0), qx);
    h3_unpack4(*(const uint2 *)(qkv + k_base + d0), kx);
    const uint2 vraw = *(const uint2 *)(qkv + v_base + d0);
    float q_sum = 0.0f, k_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_sum = fmaf(qx[i], qx[i], q_sum); k_sum = fmaf(kx[i], kx[i], k_sum);
    }
    for (uint distance = 16; distance; distance >>= 1) {
        q_sum += __shfl_down_sync(0xffffffff, q_sum, distance);
        k_sum += __shfl_down_sync(0xffffffff, k_sum, distance);
    }
    q_sum = __shfl_sync(0xffffffff, q_sum, 0);
    k_sum = __shfl_sync(0xffffffff, k_sum, 0);
    const float q_inverse = rsqrtf(q_sum / float(args.head_dim) + args.epsilon);
    const float k_inverse = rsqrtf(k_sum / float(args.head_dim) + args.epsilon);
    float qw[4], kw[4], q0[4], k0[4];
    h3_unpack4(*(const uint2 *)(q_weight + d0), qw);
    h3_unpack4(*(const uint2 *)(k_weight + d0), kw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q0[i] = bf16_round(qx[i] * q_inverse * qw[i]);
        k0[i] = bf16_round(kx[i] * k_inverse * kw[i]);
    }
    // Lanes [0, quarter) rotate with lanes [quarter, 2 quarter); same slot i.
    const uint quarter = args.rope_half / 4;
    const uint partner = lane < quarter ? lane + quarter : lane < 2 * quarter ? lane - quarter : lane;
    float q1[4], k1[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q1[i] = __shfl_sync(0xffffffff, q0[i], partner);
        k1[i] = __shfl_sync(0xffffffff, k0[i], partner);
    }
    if (lane < 2 * quarter) {
        const bool first = lane < quarter;
        const uint angle = row * args.rope_half + (first ? d0 : d0 - args.rope_half);
        float c[4], s[4];
        h3_unpack4(*(const uint2 *)(rope_cos + angle), c);
        h3_unpack4(*(const uint2 *)(rope_sin + angle), s);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (first) {
                q0[i] = bf16_round(q0[i] * c[i]) - bf16_round(q1[i] * s[i]);
                k0[i] = bf16_round(k0[i] * c[i]) - bf16_round(k1[i] * s[i]);
            } else {
                q0[i] = bf16_round(q0[i] * c[i]) + bf16_round(q1[i] * s[i]);
                k0[i] = bf16_round(k0[i] * c[i]) + bf16_round(k1[i] * s[i]);
            }
        }
    }
    ushort qo[4], ko[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        qo[i] = h3_f32_to_bf16(q0[i]);
        ko[i] = h3_f32_to_bf16(k0[i]);
    }
    const uint output_index = (row * args.heads + head) * args.head_dim + d0;
    *(uint2 *)(query + output_index) = h3_pack4(qo);
    *(uint2 *)(key + output_index) = h3_pack4(ko);
    *(uint2 *)(value + output_index) = vraw;
}
