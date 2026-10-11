// Video VAE output on the GPU: patch rows to tile pixels, tile stitching,
// temporal blend, pixel normalization and RGB24 conversion. Each kernel does
// the same float operations, in the same order, as the host code it replaces
// in h3_video_vae.c (built with --fmad=false and IEEE division), so the bytes
// are unchanged. Included at the end of h3_gpu_cuda.cu.

// Element count and dtype only: these tensors may belong to another h3_gpu
// on the same device (the decoder and its output stream share device 0).
static bool post_tensor(h3_gpu *g, const h3_gpu_tensor *t, size_t n, h3_gpu_dtype dtype,
                        const char *what) {
    if (t && t->elements >= n && t->dtype == dtype) return true;
    h3_gpu_set_error(g, "invalid tensor: %s (need %zu elements)", what, n);
    return false;
}

extern "C" int h3_gpu_event_record(h3_gpu *g, h3_gpu_event *e) {
    return e && cuda_ok(g, cudaEventRecord(e->event, g->stream));
}

// Copy between devices on the source GPU's stream (NVLink peer copy).
extern "C" int h3_gpu_copy_peer_f32(h3_gpu *g, h3_gpu_tensor *dst, size_t dst_offset,
                                    const h3_gpu_tensor *src, size_t src_offset, size_t count) {
    if (!post_tensor(g, dst, dst_offset + count, H3_GPU_F32, "peer copy destination") ||
        !post_tensor(g, src, src_offset + count, H3_GPU_F32, "peer copy source"))
        return 0;
    int src_device = 0, dst_device = 0;
    cudaPointerAttributes a{}, b{};
    if (!cuda_ok(g, cudaPointerGetAttributes(&a, src->data)) ||
        !cuda_ok(g, cudaPointerGetAttributes(&b, dst->data)))
        return 0;
    src_device = a.device;
    dst_device = b.device;
    return cuda_ok(g, cudaMemcpyPeerAsync((float *)dst->data + dst_offset, dst_device,
                                          (const float *)src->data + src_offset, src_device,
                                          count * sizeof(float), g->stream));
}

// Synchronous device-to-host copy of `count` elements from `offset`.
extern "C" int h3_gpu_download(h3_gpu *g, const h3_gpu_tensor *t, size_t offset, void *host,
                               size_t count) {
    size_t size = t ? t->bytes / std::max<size_t>(t->elements, 1) : 0;
    if (!t || !host || offset + count > t->elements) {
        h3_gpu_set_error(g, "invalid download range");
        return 0;
    }
    return cuda_ok(g, cudaMemcpyAsync(host, (const char *)t->data + offset * size, count * size,
                                      cudaMemcpyDeviceToHost, g->stream)) &&
           cuda_ok(g, cudaStreamSynchronize(g->stream));
}

// unpack_rows: frame f of the tile reads decoded time f + offset, skipping
// `gap` decoded frames from frame `gap_from` on.
__global__ void vae_unpack_tile(float *tile, const float *rows, int latent_h, int latent_w,
                                int first_frame, int frame_count, int frame_offset,
                                int gap_from, int gap, int output_patch) {
    int pixel_h = latent_h * 16, pixel_w = latent_w * 16;
    size_t total = (size_t)frame_count * pixel_h * pixel_w * 3;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        int channel = int(i % 3);
        size_t pixel = i / 3;
        int x = int(pixel % pixel_w), y = int(pixel / pixel_w % pixel_h);
        int output_frame = int(pixel / ((size_t)pixel_w * pixel_h));
        int frame = first_frame + output_frame;
        int decoded_t = frame + frame_offset + (frame >= gap_from ? gap : 0);
        size_t patch = ((size_t)(decoded_t / 4) * latent_h + y / 16) * latent_w + x / 16;
        size_t component = (((size_t)channel * 4 + decoded_t % 4) * 16 + y % 16) * 16 + x % 16;
        tile[i] = rows[patch * output_patch + component];
    }
}

extern "C" int h3_gpu_vae_unpack_tile_f32(h3_gpu *g, h3_gpu_tensor *dst, size_t dst_offset,
                                          const h3_gpu_tensor *rows, int latent_h, int latent_w,
                                          int first_frame, int frame_count, int frame_offset,
                                          int gap_from, int gap, int output_patch) {
    size_t count = (size_t)frame_count * latent_h * 16 * latent_w * 16 * 3;
    if (!post_tensor(g, dst, dst_offset + count, H3_GPU_F32, "VAE tile pixels") ||
        !post_tensor(g, rows, 1, H3_GPU_F32, "VAE output patches"))
        return 0;
    uint blocks = uint(std::min<size_t>((count + 255) / 256, 65535));
    vae_unpack_tile<<<blocks, 256, 0, g->stream>>>((float *)dst->data + dst_offset,
        (const float *)rows->data, latent_h, latent_w, first_frame, frame_count, frame_offset,
        gap_from, gap, output_patch);
    return launch_ok(g);
}

// stitch_tile_frames: the region a tile owns, blended with the tile above
// and then the tile to the left.
__global__ void vae_stitch_tile(float *rgb, const float *current, const float *above,
                                const float *left, h3_vae_stitch s) {
    size_t total = (size_t)s.frames * s.keep_h * s.keep_w * 3;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        int channel = int(i % 3);
        size_t pixel = i / 3;
        int x = int(pixel % s.keep_w), y = int(pixel / s.keep_w % s.keep_h);
        int frame = int(pixel / ((size_t)s.keep_w * s.keep_h));
        size_t local = (((size_t)frame * s.length_y + y) * s.length_x + x) * 3 + channel;
        float value = current[local];
        if (above && y < s.overlap_y) {
            size_t top = (((size_t)frame * s.length_y + (s.length_y - s.overlap_y + y)) *
                          s.length_x + x) * 3 + channel;
            float alpha = (float)y / (float)s.overlap_y;
            value = above[top] * (1.0f - alpha) + value * alpha;
        }
        if (left && x < s.overlap_x) {
            size_t prior = (((size_t)frame * s.length_y + y) * s.length_x +
                            (s.length_x - s.overlap_x + x)) * 3 + channel;
            float alpha = (float)x / (float)s.overlap_x;
            value = left[prior] * (1.0f - alpha) + value * alpha;
        }
        size_t destination = (((size_t)frame * s.full_h + (s.start_y + y)) * s.full_w +
                              (s.start_x + x)) * 3 + channel;
        rgb[destination] = value;
    }
}

extern "C" int h3_gpu_vae_stitch_tile_f32(h3_gpu *g, h3_gpu_tensor *canvas,
                                          const h3_gpu_tensor *store, size_t current,
                                          long long above, long long left,
                                          const h3_vae_stitch *s) {
    if (!s) return 0;
    size_t tile = (size_t)s->frames * s->length_y * s->length_x * 3;
    size_t count = (size_t)s->frames * s->keep_h * s->keep_w * 3;
    if (!post_tensor(g, canvas, (size_t)s->frames * s->full_h * s->full_w * 3,
                           H3_GPU_F32, "VAE canvas") ||
        !post_tensor(g, store, current + tile, H3_GPU_F32, "VAE tile store") ||
        (above >= 0 && !post_tensor(g, store, size_t(above) + tile, H3_GPU_F32, "tile above")) ||
        (left >= 0 && !post_tensor(g, store, size_t(left) + tile, H3_GPU_F32, "tile left")))
        return 0;
    const float *base = (const float *)store->data;
    uint blocks = uint(std::min<size_t>((count + 255) / 256, 65535));
    vae_stitch_tile<<<blocks, 256, 0, g->stream>>>((float *)canvas->data, base + current,
        above >= 0 ? base + above : nullptr, left >= 0 ? base + left : nullptr, *s);
    return launch_ok(g);
}

// temporal_blend_range: frame f of the chunk weighs itself by f / 5.
__global__ void vae_temporal_blend(float *rgb, const float *overlap, size_t frame_elements,
                                   size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        float alpha = (float)(i / frame_elements) / 5.0f;
        rgb[i] = overlap[i] * (1.0f - alpha) + rgb[i] * alpha;
    }
}

extern "C" int h3_gpu_vae_temporal_blend_f32(h3_gpu *g, h3_gpu_tensor *rgb,
                                             const h3_gpu_tensor *overlap, size_t frame_elements,
                                             int frames) {
    size_t total = frame_elements * frames;
    if (!post_tensor(g, rgb, total, H3_GPU_F32, "VAE blend frames") ||
        !post_tensor(g, overlap, total, H3_GPU_F32, "VAE blend overlap"))
        return 0;
    uint blocks = uint(std::min<size_t>((total + 255) / 256, 65535));
    vae_temporal_blend<<<blocks, 256, 0, g->stream>>>((float *)rgb->data,
        (const float *)overlap->data, frame_elements, total);
    return launch_ok(g);
}

// finalize_pixels: x * std + mean, clipped to [0, 1]; channel = index % 3.
__global__ void vae_finalize(float *rgb, size_t count) {
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float deviation[] = {0.229f, 0.224f, 0.225f};
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        float value = rgb[i] * deviation[i % 3] + mean[i % 3];
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        rgb[i] = value;
    }
}

extern "C" int h3_gpu_vae_finalize_f32(h3_gpu *g, h3_gpu_tensor *rgb, size_t offset,
                                       size_t count) {
    if (offset % 3 || !post_tensor(g, rgb, offset + count, H3_GPU_F32, "VAE pixels")) return 0;
    uint blocks = uint(std::min<size_t>((count + 255) / 256, 65535));
    vae_finalize<<<blocks, 256, 0, g->stream>>>((float *)rgb->data + offset, count);
    return launch_ok(g);
}

// rgb8_range: lrintf(clamp(x * 255)), round half to even.
__global__ void vae_rgb8(unsigned char *out, const float *rgb, size_t count) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        float scaled = rgb[i] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        out[i] = (unsigned char)__float2int_rn(scaled);
    }
}

extern "C" int h3_gpu_vae_rgb8(h3_gpu *g, h3_gpu_tensor *out, const h3_gpu_tensor *rgb,
                               size_t offset, size_t count) {
    if (!post_tensor(g, out, count, H3_GPU_I8, "RGB24 output") ||
        !post_tensor(g, rgb, offset + count, H3_GPU_F32, "VAE pixels"))
        return 0;
    uint blocks = uint(std::min<size_t>((count + 255) / 256, 65535));
    vae_rgb8<<<blocks, 256, 0, g->stream>>>((unsigned char *)out->data,
                                             (const float *)rgb->data + offset, count);
    return launch_ok(g);
}

// Lets the current device read and write `peer` directly (NVLink). Without
// it peer copies still work, staged through host memory.
extern "C" void h3_gpu_enable_peer_access(int peer) {
    int can = 0, device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device == peer ||
        cudaDeviceCanAccessPeer(&can, device, peer) != cudaSuccess || !can)
        return;
    if (cudaDeviceEnablePeerAccess(peer, 0) != cudaSuccess) cudaGetLastError();
}

// h3_euler_velocity_step (CUDA branch), on the patch rows of the sampler:
// denoised = x + sigma' * v; x = r * x + (1 - r) * denoised, with
// sigma' = 1 - (1 - sigma) and r = sigma_next / sigma, as on the host.
__global__ void euler_velocity(float *sample, const float *velocity, size_t count,
                               float sigma, float sigma_next) {
    float timestep = 1.0f - sigma;
    float sigma_from_timestep = 1.0f - timestep;
    float ratio = sigma_next / sigma;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        float denoised = sample[i] + sigma_from_timestep * velocity[i];
        sample[i] = ratio * sample[i] + (1.0f - ratio) * denoised;
    }
}

extern "C" int h3_gpu_euler_velocity_f32(h3_gpu *g, h3_gpu_tensor *sample, size_t offset,
                                         const h3_gpu_tensor *velocity, size_t count,
                                         float sigma, float sigma_next) {
    if (!post_tensor(g, sample, offset + count, H3_GPU_F32, "Euler sample") ||
        !post_tensor(g, velocity, count, H3_GPU_F32, "Euler velocity"))
        return 0;
    uint blocks = uint(std::min<size_t>((count + 255) / 256, 65535));
    euler_velocity<<<blocks, 256, 0, g->stream>>>((float *)sample->data + offset,
        (const float *)velocity->data, count, sigma, sigma_next);
    return launch_ok(g);
}
