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
#ifdef __cplusplus
}
#endif
#endif
