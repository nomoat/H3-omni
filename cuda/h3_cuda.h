#ifndef H3_CUDA_H
#define H3_CUDA_H
#include "h3_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Qwen rounds the normalized activation before multiplying by its BF16 weight. */
int h3_gpu_qwen_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight, uint32_t rows,
                              uint32_t width, float epsilon);
/* Project contiguous groups with the reference's per-evaluation batch shapes.
 * row_counts is a host array; its nonzero entries must sum to rows. */
int h3_gpu_time_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                         const h3_gpu_tensor *bias, uint32_t rows,
                         uint32_t input_dim, uint32_t output_dim,
                         const uint32_t *row_counts, uint32_t group_count);
int h3_gpu_time_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t input_dim, uint32_t output_dim,
                          const uint32_t *row_counts, uint32_t group_count);
int h3_gpu_time_features_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *timesteps, uint32_t rows,
                           uint32_t width);
int h3_gpu_final_heads_f32(h3_gpu *gpu, h3_gpu_tensor *video_output,
                         h3_gpu_tensor *audio_output, const h3_gpu_tensor *hidden,
                         const h3_gpu_tensor *norm, const h3_gpu_tensor *modulation,
                         const h3_gpu_tensor *row_map, const h3_gpu_tensor *video_weight,
                         const h3_gpu_tensor *video_bias, const h3_gpu_tensor *audio_weight,
                         const h3_gpu_tensor *audio_bias, uint32_t rows, uint32_t width,
                         uint32_t video_start, uint32_t video_rows,
                         uint32_t audio_start, uint32_t audio_rows);
/* Split an FP32 weight into a TF32-exact high part (kept in place) and a
 * low remainder stored beside it. FP32 linears with a split weight then run
 * as three TF32 tensor-core products, xh*wh + xh*wl + xl*wh (3xTF32). */
int h3_gpu_tensor_split_tf32(h3_gpu *gpu, h3_gpu_tensor *weight);
int h3_gpu_tensor_is_split_tf32(const h3_gpu_tensor *weight);

/* Asynchronous transfers for pipelined video VAE tiles (cuda/vae_fused.cuh). */
typedef struct h3_gpu_event h3_gpu_event;
h3_gpu_event *h3_gpu_event_new(h3_gpu *gpu);
void h3_gpu_event_free(h3_gpu_event *event);
/* Blocks the calling host thread; safe from a thread other than the GPU user. */
int h3_gpu_event_wait(h3_gpu_event *event, char *error, size_t error_size);
void *h3_gpu_host_alloc(size_t bytes); /* page-locked */
void h3_gpu_host_free(void *pointer);
int h3_gpu_tensor_upload_f32_async(h3_gpu *gpu, h3_gpu_tensor *tensor,
                                   const float *host, size_t count);
int h3_gpu_tensor_download_f32_async(h3_gpu *gpu, const h3_gpu_tensor *tensor,
                                     float *host, size_t count,
                                     h3_gpu_event *done);
int h3_gpu_stream_wait_event(h3_gpu *gpu, h3_gpu_event *event);

/* 3xTF32 video VAE fusions; same values as the unfused operations.
 * "parts" tensors hold TF32 high parts followed by low remainders. */
int h3_gpu_rms_norm_split_tf32(h3_gpu *gpu, h3_gpu_tensor *parts,
                               const h3_gpu_tensor *input,
                               const h3_gpu_tensor *weight, uint32_t rows,
                               uint32_t width, float epsilon);
int h3_gpu_linear_split_input_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *parts,
                                  const h3_gpu_tensor *weight, uint32_t rows,
                                  uint32_t input_dim, uint32_t output_dim);
int h3_gpu_video_qkv_rope_bias_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                                   h3_gpu_tensor *key, h3_gpu_tensor *value,
                                   const h3_gpu_tensor *qkv,
                                   const h3_gpu_tensor *bias,
                                   const h3_gpu_tensor *rope_cos,
                                   const h3_gpu_tensor *rope_sin,
                                   uint32_t sequence, uint32_t heads,
                                   uint32_t head_dim, uint32_t rope_half,
                                   float epsilon);
int h3_gpu_swiglu_bias_split_tf32(h3_gpu *gpu, h3_gpu_tensor *parts,
                                  const h3_gpu_tensor *fused,
                                  const h3_gpu_tensor *bias, uint32_t rows,
                                  uint32_t width);
int h3_gpu_scale_add_bias_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *residual,
                              const h3_gpu_tensor *branch,
                              const h3_gpu_tensor *bias,
                              const h3_gpu_tensor *scale, uint32_t rows,
                              uint32_t width);
#ifdef __cplusplus
}
#endif
#endif
