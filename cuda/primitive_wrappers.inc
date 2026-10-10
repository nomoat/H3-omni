// Generated from h3_gpu.m.
int h3_gpu_silu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, input, elements, "SiLU input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "SiLU output") ||
        output->dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_silu_f32", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &elements, sizeof(elements));
        });
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, input, elements, "cast input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "cast output") ||
        output->dtype != H3_GPU_BF16) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_cast_f32_to_bf16", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &elements, sizeof(elements));
        });
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, input, elements, "cast input") ||
        input->dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, output, elements, "cast output") ||
        output->dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_cast_bf16_to_f32", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &elements, sizeof(elements));
        });
}

int h3_gpu_rms_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, "RMSNorm input") ||
        !h3_gpu_require_elements(gpu, weight, width, "RMSNorm weight") ||
        !h3_gpu_require_elements(gpu, output, count, "RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, "h3_rms_norm_f32", rows,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, weight, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &args, sizeof(args));
        });
}

int h3_gpu_adaln_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, "AdaLN input") ||
        !h3_gpu_require_elements(gpu, norm_weight, width, "AdaLN norm") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "AdaLN row map") ||
        !h3_gpu_require_elements(gpu, output, count, "AdaLN output") ||
        !modulation || shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_2d(gpu, "h3_adaln_f32", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, norm_weight, 0);
            encoder.buffer(2, modulation, 0);
            encoder.buffer(3, row_map, 0);
            encoder.buffer(4, output, 0);
            encoder.arg(5, &args, sizeof(args));
        });
}

int h3_gpu_gate_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, "gate residual") ||
        !h3_gpu_require_elements(gpu, branch, count, "gate branch") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "gate row map") ||
        !h3_gpu_require_elements(gpu, output, count, "gate output") ||
        !modulation || gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, "h3_gate_f32", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, residual, 0);
            encoder.buffer(1, branch, 0);
            encoder.buffer(2, modulation, 0);
            encoder.buffer(3, row_map, 0);
            encoder.buffer(4, output, 0);
            encoder.arg(5, &args, sizeof(args));
        });
}

int h3_gpu_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, "QKV input") ||
        !h3_gpu_require_elements(gpu, q_norm, head_dim, "Q norm") ||
        !h3_gpu_require_elements(gpu, k_norm, head_dim, "K norm") ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, "RoPE cosine") ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, "RoPE sine") ||
        !h3_gpu_require_elements(gpu, query, count, "query") ||
        !h3_gpu_require_elements(gpu, key, count, "key") ||
        !h3_gpu_require_elements(gpu, value, count, "value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, "h3_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        [&](Binding &encoder) {
            encoder.buffer(0, qkv, 0);
            encoder.buffer(1, q_norm, 0);
            encoder.buffer(2, k_norm, 0);
            encoder.buffer(3, rope_cos, 0);
            encoder.buffer(4, rope_sin, 0);
            encoder.buffer(5, query, 0);
            encoder.buffer(6, key, 0);
            encoder.buffer(7, value, 0);
            encoder.arg(8, &args, sizeof(args));
        });
}

int h3_gpu_swiglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, "SwiGLU input") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * width, "SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, "h3_swiglu_f32", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, fused, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}

int h3_gpu_scale_add_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, "scale-add residual") ||
        residual->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, branch, count, "scale-add branch") ||
        branch->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, scale, width, "scale-add scale") ||
        scale->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "scale-add output") ||
        output->dtype != H3_GPU_F32) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, "h3_scale_add_f32", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, residual, 0);
            encoder.buffer(1, branch, 0);
            encoder.buffer(2, scale, 0);
            encoder.buffer(3, output, 0);
            encoder.arg(4, &args, sizeof(args));
        });
}

int h3_gpu_layer_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, "LayerNorm input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, width, "LayerNorm weight") ||
        weight->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, width, "LayerNorm bias") ||
        bias->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "LayerNorm output") ||
        output->dtype != H3_GPU_F32) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, "h3_layer_norm_f32", rows,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, weight, 0);
            encoder.buffer(2, bias, 0);
            encoder.buffer(3, output, 0);
            encoder.arg(4, &args, sizeof(args));
        });
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    h3_gpu *gpu = opaque;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, "video QKV") ||
        qkv->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, "video RoPE cosine") ||
        rope_cos->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, "video RoPE sine") ||
        rope_sin->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, "video query") ||
        query->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, key, count, "video key") ||
        key->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, value, count, "video value") ||
        value->dtype != H3_GPU_F32 || rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, "h3_video_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        [&](Binding &encoder) {
            encoder.buffer(0, qkv, 0);
            encoder.buffer(1, rope_cos, 0);
            encoder.buffer(2, rope_sin, 0);
            encoder.buffer(3, query, 0);
            encoder.buffer(4, key, 0);
            encoder.buffer(5, value, 0);
            encoder.arg(6, &args, sizeof(args));
        });
}

int h3_gpu_weight_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)outer * inner;
    if (!outer || !inner ||
        !h3_gpu_require_elements(gpu, vector, count, "weight-norm vector") ||
        vector->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, magnitude, outer,
                                 "weight-norm magnitude") ||
        magnitude->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "weight-norm output") ||
        output->dtype != H3_GPU_F32) return 0;
    weight_norm_args args = {outer, inner};
    return h3_gpu_dispatch_1d(gpu, "h3_weight_norm_f32", outer,
        [&](Binding &encoder) {
            encoder.buffer(0, vector, 0);
            encoder.buffer(1, magnitude, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &args, sizeof(args));
        });
}

int h3_gpu_add_scaled_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, left, elements, "scaled-add left") ||
        left->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, right, elements, "scaled-add right") ||
        right->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "scaled-add output") ||
        output->dtype != H3_GPU_F32) return 0;
    add_scaled_args args = {elements, left_scale, right_scale};
    return h3_gpu_dispatch_1d(gpu, "h3_add_scaled_f32", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, left, 0);
            encoder.buffer(1, right, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &args, sizeof(args));
        });
}

int h3_gpu_alias_free_snake_f32(
                          h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels ||
        !h3_gpu_require_elements(gpu, input, count, "Snake input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "Snake output") ||
        output->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha_log, channels, "Snake alpha") ||
        alpha_log->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, beta_log, channels, "Snake beta") ||
        beta_log->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, upsample_filter, 12,
                                 "Snake upsample filter") ||
        upsample_filter->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, downsample_filter, 12,
                                 "Snake downsample filter") ||
        downsample_filter->dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_3d(gpu, "h3_alias_free_snake_f32",
        MTLSizeMake(channels, length, batch),
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, alpha_log, 0);
            encoder.buffer(2, beta_log, 0);
            encoder.buffer(3, upsample_filter, 0);
            encoder.buffer(4, downsample_filter, 0);
            encoder.buffer(5, output, 0);
            encoder.arg(6, &args, sizeof(args));
        });
}

int h3_gpu_snake1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, input, count, "Snake1d input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha, channels, "Snake1d alpha") ||
        alpha->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, "Snake1d output") ||
        output->dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_1d(gpu, "h3_snake1d_f32", (uint32_t)count,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, alpha, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &args, sizeof(args));
        });
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *opaque,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    h3_gpu *gpu = opaque;
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!batch || !length || !heads || !head_dim || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, qkv, count * 3, "audio QKV") ||
        qkv->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, q_bias, width, "audio Q bias") ||
        !h3_gpu_require_elements(gpu, k_bias, width, "audio K bias") ||
        !h3_gpu_require_elements(gpu, v_bias, width, "audio V bias") ||
        q_bias->dtype != H3_GPU_F32 ||
        k_bias->dtype != H3_GPU_F32 ||
        v_bias->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, "audio query") ||
        !h3_gpu_require_elements(gpu, key, count, "audio key") ||
        !h3_gpu_require_elements(gpu, value, count, "audio value") ||
        query->dtype != H3_GPU_F32 || key->dtype != H3_GPU_F32 ||
        value->dtype != H3_GPU_F32) return 0;
    audio_qkv_args args = {batch, length, heads, head_dim};
    return h3_gpu_dispatch_1d(gpu, "h3_audio_qkv_split_f32",
        (uint32_t)count, [&](Binding &encoder) {
            encoder.buffer(0, qkv, 0);
            encoder.buffer(1, q_bias, 0);
            encoder.buffer(2, k_bias, 0);
            encoder.buffer(3, v_bias, 0);
            encoder.buffer(4, query, 0);
            encoder.buffer(5, key, 0);
            encoder.buffer(6, value, 0);
            encoder.arg(7, &args, sizeof(args));
        });
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *opaque,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    h3_gpu *gpu = opaque;
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, attended, input_count,
                                 "audio attended values") ||
        attended->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 "audio pooled values") ||
        output->dtype != H3_GPU_F32) return 0;
    audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    return h3_gpu_dispatch_1d(gpu, "h3_audio_attention_pool_f32",
        (uint32_t)output_count, [&](Binding &encoder) {
            encoder.buffer(0, attended, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}

int h3_gpu_geglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_elements(gpu, gate, elements, "GeGLU gate") ||
        !h3_gpu_require_elements(gpu, linear, elements, "GeGLU linear") ||
        !h3_gpu_require_elements(gpu, output, elements, "GeGLU output") ||
        gate->dtype != H3_GPU_F32 ||
        linear->dtype != H3_GPU_F32 ||
        output->dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_geglu_f32", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, gate, 0);
            encoder.buffer(1, linear, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &elements, sizeof(elements));
        });
}

int h3_gpu_clip_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    h3_gpu *gpu = opaque;
    if (!(minimum <= maximum) ||
        !h3_gpu_require_elements(gpu, input, elements, "clip input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, "clip output") ||
        output->dtype != H3_GPU_F32) return 0;
    clip_args args = {elements, minimum, maximum};
    return h3_gpu_dispatch_1d(gpu, "h3_clip_f32", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    h3_gpu *gpu = opaque;
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 "VAE encoder pad input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 "VAE encoder pad output") ||
        output->dtype != H3_GPU_F32) return 0;
    vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    return h3_gpu_dispatch_3d(gpu, "h3_vae_encoder_pad_f32",
        MTLSizeMake(channels, output_width,
                    (size_t)batch * output_depth * output_height),
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!batch || !depth || !height || !width || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f) ||
        !h3_gpu_require_elements(gpu, input, count,
                                 "VAE encoder norm input") ||
        input->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, channels,
                                 "VAE encoder norm weight") ||
        weight->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, channels,
                                 "VAE encoder norm bias") ||
        bias->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count,
                                 "VAE encoder norm output") ||
        output->dtype != H3_GPU_F32) return 0;
    vae_encoder_norm_args args = {
        batch, depth, height, width, channels, groups, epsilon
    };
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    return h3_gpu_dispatch_rows(
        gpu, "h3_vae_encoder_group_norm_silu_f32", (uint32_t)rows,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, weight, 0);
            encoder.buffer(2, bias, 0);
            encoder.buffer(3, output, 0);
            encoder.arg(4, &args, sizeof(args));
        });
}

int h3_gpu_silu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, input, elements, "SiLU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, "SiLU output")) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_silu_bf16", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &elements, sizeof(elements));
        });
}

int h3_gpu_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, "RMSNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, "RMSNorm weight") ||
        !h3_gpu_require_bf16(gpu, output, count, "RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, "h3_rms_norm_bf16", rows,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, weight, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &args, sizeof(args));
        });
}

int h3_gpu_layer_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, "LayerNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, "LayerNorm weight") ||
        !h3_gpu_require_bf16(gpu, bias, width, "LayerNorm bias") ||
        !h3_gpu_require_bf16(gpu, output, count, "LayerNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, "h3_layer_norm_bf16", rows,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, weight, 0);
            encoder.buffer(2, bias, 0);
            encoder.buffer(3, output, 0);
            encoder.arg(4, &args, sizeof(args));
        });
}

int h3_gpu_gelu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, input, elements, "GELU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, "GELU output")) return 0;
    gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    return h3_gpu_dispatch_1d(gpu, "h3_gelu_bf16", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    h3_gpu *gpu = opaque;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, "vision QKV") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                              "vision RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                              "vision RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, "vision query") ||
        !h3_gpu_require_bf16(gpu, key, count, "vision key") ||
        !h3_gpu_require_bf16(gpu, value, count, "vision value") ||
        rope_half * 2 != head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f};
    return h3_gpu_dispatch_3d(gpu, "h3_vision_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        [&](Binding &encoder) {
            encoder.buffer(0, qkv, 0);
            encoder.buffer(1, rope_cos, 0);
            encoder.buffer(2, rope_sin, 0);
            encoder.buffer(3, query, 0);
            encoder.buffer(4, key, 0);
            encoder.buffer(5, value, 0);
            encoder.arg(6, &args, sizeof(args));
        });
}

int h3_gpu_gate_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    h3_gpu *gpu = opaque;
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, residual, count, "gate residual") ||
        !h3_gpu_require_bf16(gpu, branch, count, "gate branch") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, "gate modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, "gate row map") ||
        row_map->dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, "gate output") ||
        gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
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

int h3_gpu_swiglu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, fused, (size_t)rows * width * 2,
                              "SwiGLU input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width,
                              "SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, "h3_swiglu_bf16", width, rows,
        [&](Binding &encoder) {
            encoder.buffer(0, fused, 0);
            encoder.buffer(1, output, 0);
            encoder.arg(2, &args, sizeof(args));
        });
}

int h3_gpu_embedding_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, weight, (size_t)vocab_size * width,
                              "embedding weight") ||
        !h3_gpu_require_elements(gpu, token_ids, tokens, "token IDs") ||
        token_ids->dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, (size_t)tokens * width,
                              "embedding output")) return 0;
    embedding_args args = {tokens, vocab_size, width};
    return h3_gpu_dispatch_2d(gpu, "h3_embedding_bf16", width, tokens,
        [&](Binding &encoder) {
            encoder.buffer(0, weight, 0);
            encoder.buffer(1, token_ids, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &args, sizeof(args));
        });
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *opaque,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    h3_gpu *gpu = opaque;
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query_input, query_count, "text query") ||
        !h3_gpu_require_bf16(gpu, key_input, key_count, "text key") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, "text Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, "text K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, "text RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, "text RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query_output, query_count, "text query output") ||
        !h3_gpu_require_bf16(gpu, key_output, key_count, "text key output")) return 0;
    text_rope_args args = {sequence, query_heads, kv_heads, head_dim, epsilon};
    return h3_gpu_dispatch_3d(gpu, "h3_text_qk_rope_bf16",
        MTLSizeMake(head_dim, query_heads, sequence),
        [&](Binding &encoder) {
            encoder.buffer(0, query_input, 0);
            encoder.buffer(1, key_input, 0);
            encoder.buffer(2, q_norm, 0);
            encoder.buffer(3, k_norm, 0);
            encoder.buffer(4, rope_cos, 0);
            encoder.buffer(5, rope_sin, 0);
            encoder.buffer(6, query_output, 0);
            encoder.buffer(7, key_output, 0);
            encoder.arg(8, &args, sizeof(args));
        });
}

int h3_gpu_rope_text_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    h3_gpu *gpu = opaque;
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query, query_count, "RoPE query") ||
        !h3_gpu_require_bf16(gpu, key, key_count, "RoPE key") ||
        !h3_gpu_require_elements(gpu, rope_cos_f32, rope_count, "RoPE cosine") ||
        rope_cos_f32->dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin_f32, rope_count, "RoPE sine") ||
        rope_sin_f32->dtype != H3_GPU_F32) return 0;
    text_rope_inplace_args args = {sequence, query_heads, kv_heads, head_dim};
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    return h3_gpu_dispatch_2d(gpu, "h3_rope_text_bf16", sequence, maximum_heads,
        [&](Binding &encoder) {
            encoder.buffer(0, query, 0);
            encoder.buffer(1, key, 0);
            encoder.buffer(2, rope_cos_f32, 0);
            encoder.buffer(3, rope_sin_f32, 0);
            encoder.arg(4, &args, sizeof(args));
        });
}

int h3_gpu_add_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, left, elements, "add left") ||
        !h3_gpu_require_bf16(gpu, right, elements, "add right") ||
        !h3_gpu_require_bf16(gpu, output, elements, "add output")) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_add_bf16", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, left, 0);
            encoder.buffer(1, right, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &elements, sizeof(elements));
        });
}

int h3_gpu_sub_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, left, elements, "subtract left") ||
        !h3_gpu_require_bf16(gpu, right, elements, "subtract right") ||
        !h3_gpu_require_bf16(gpu, output, elements, "subtract output"))
        return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_sub_bf16", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, left, 0);
            encoder.buffer(1, right, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &elements, sizeof(elements));
        });
}

int h3_gpu_silu_mul_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    h3_gpu *gpu = opaque;
    if (!h3_gpu_require_bf16(gpu, gate, elements, "SiLU gate") ||
        !h3_gpu_require_bf16(gpu, up, elements, "SiLU up") ||
        !h3_gpu_require_bf16(gpu, output, elements, "SiLU product")) return 0;
    return h3_gpu_dispatch_1d(gpu, "h3_silu_mul_bf16", elements,
        [&](Binding &encoder) {
            encoder.buffer(0, gate, 0);
            encoder.buffer(1, up, 0);
            encoder.buffer(2, output, 0);
            encoder.arg(3, &elements, sizeof(elements));
        });
}
