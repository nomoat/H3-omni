#ifndef H3_CUDA_H
#define H3_CUDA_H
#include "h3_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Multi-GPU (cuda/parallel.cuh). A thread selects its device before it
 * creates or uses an h3_gpu. One NCCL communicator per device 0..count-1.
 * Ulysses: scatter_heads turns this rank's [rows][heads][dim] rows into all
 * ranks' rows for heads/ranks heads, [ranks*rows][heads/ranks][dim];
 * gather_heads is the inverse. Values are moved, never changed. */
typedef struct h3_gpu_comm h3_gpu_comm;
int h3_gpu_device_count(void);
int h3_gpu_select_device(int device, char *error, size_t error_size);
int h3_gpu_comm_init_all(int count, h3_gpu_comm **comms, char *error, size_t error_size);
void h3_gpu_comm_free(h3_gpu_comm *comm);
void h3_gpu_comm_abort(h3_gpu_comm *comm);
int h3_gpu_ulysses_scatter_heads_bf16(h3_gpu *gpu, h3_gpu_comm *comm, h3_gpu_tensor *tensor,
                                      h3_gpu_tensor *scratch, uint32_t rows, uint32_t heads,
                                      uint32_t dim);
int h3_gpu_ulysses_gather_heads_bf16(h3_gpu *gpu, h3_gpu_comm *comm, h3_gpu_tensor *tensor,
                                     h3_gpu_tensor *scratch, uint32_t rows, uint32_t heads,
                                     uint32_t dim);
int h3_gpu_zero(h3_gpu *gpu, h3_gpu_tensor *tensor);
int h3_gpu_all_gather_bf16(h3_gpu *gpu, h3_gpu_comm *comm, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t count);
typedef struct h3_gpu_vsa h3_gpu_vsa;
h3_gpu_vsa *h3_gpu_vsa_new(h3_gpu *gpu, uint32_t text_rows, uint32_t audio_rows,
                           uint32_t time, uint32_t height, uint32_t width, uint32_t heads);
void h3_gpu_vsa_free(h3_gpu_vsa *vsa);
int h3_gpu_vsa_bf16(h3_gpu_vsa *vsa, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, const h3_gpu_tensor *gate);
/* FastH3 text refinement needs FA2 rounding, independent of H3_FLASH3. */
int h3_gpu_sdpa_fa2_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                         const h3_gpu_tensor *value, uint32_t tokens,
                         uint32_t heads, uint32_t head_dim, float scale);
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

/* Non-causal FP32 attention, head dim 64, [token][head][64] layout: K and V
 * are split into TF32 high/low parts, then one fused kernel computes both
 * products as 3xTF32 tensor-core products with an online FP32 softmax.
 * Error against FP64 is the same as h3_gpu_sdpa_f32 (about 1e-6), but it
 * rounds differently (video VAE default mode only; cuda/vae_attention.cuh). */
int h3_gpu_sdpa_3xtf32_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t tokens,
                           uint32_t heads, uint32_t head_dim, float scale);

/* Video VAE FP16 mode (cuda/vae_fp16.cuh): FP16 tensor-core products with
 * FP32 accumulation and FP32 outputs; norms, QK norm, RoPE, SwiGLU, biases
 * and the residual stream stay FP32, only product inputs are FP16. */
h3_gpu_tensor *h3_gpu_tensor_new_f16(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_to_f16(h3_gpu *gpu, const h3_gpu_tensor *f32);
int h3_gpu_rms_norm_f16_out(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight, uint32_t rows,
                            uint32_t width, float epsilon);
int h3_gpu_video_qkv_rope_bias_f16(h3_gpu *gpu, h3_gpu_tensor *query,
                                   h3_gpu_tensor *key, h3_gpu_tensor *value,
                                   const h3_gpu_tensor *qkv,
                                   const h3_gpu_tensor *bias,
                                   const h3_gpu_tensor *rope_cos,
                                   const h3_gpu_tensor *rope_sin,
                                   uint32_t sequence, uint32_t heads,
                                   uint32_t head_dim, uint32_t rope_half,
                                   float epsilon);
int h3_gpu_scale_add_bias_rms_f16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  h3_gpu_tensor *normed, const h3_gpu_tensor *residual,
                                  const h3_gpu_tensor *branch, const h3_gpu_tensor *bias,
                                  const h3_gpu_tensor *scale, const h3_gpu_tensor *weight,
                                  uint32_t rows, uint32_t width, float eps);
int h3_gpu_swiglu_bias_f16_out(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *fused,
                               const h3_gpu_tensor *bias, uint32_t rows,
                               uint32_t width);
int h3_gpu_linear_f16_f32_out(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight, uint32_t rows,
                              uint32_t in_features, uint32_t out_features);
int h3_gpu_sdpa_f16_d64(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                        const h3_gpu_tensor *value, uint32_t tokens,
                        uint32_t heads, uint32_t head_dim, float scale);
#ifdef __cplusplus
}
#endif
#endif
