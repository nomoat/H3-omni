#include "h3_video_vae.h"

#include "h3_weights.h"
#ifdef H3_CUDA
#include "cuda/h3_cuda.h"
#endif

#include <errno.h>
#include <math.h>
#ifdef H3_CUDA
#include <pthread.h>
#endif
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 24,
    CHUNK_LATENT_TIME = 7,
    FIRST_CHUNK_FRAMES = 22,
    FRAME_OFFSET = 3,
    HIDDEN = 2048,
    LAYERS = 36,
    HEADS = 32,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    REGISTERS = 4,
    SUFFIX = 5,
    ROPE_HALF = 24,
    OUTPUT_PATCH = 3 * 4 * 16 * 16,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *qkv_w;
    h3_gpu_tensor *qkv_b;
    h3_gpu_tensor *out_w;
    h3_gpu_tensor *out_b;
    h3_gpu_tensor *scale1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *w1;
    h3_gpu_tensor *w1_b;
    h3_gpu_tensor *w2;
    h3_gpu_tensor *w2_b;
    h3_gpu_tensor *scale2;
    int tf32_attention; /* fused 3xTF32 attention (CUDA default mode) */
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    vae_block blocks[LAYERS];
    h3_gpu_tensor *post_w;
    h3_gpu_tensor *post_b;
    h3_gpu_tensor *embed_w;
    h3_gpu_tensor *embed_b;
    h3_gpu_tensor *registers;
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *norm_out_b;
    h3_gpu_tensor *proj_w;
    h3_gpu_tensor *proj_b;
    h3_gpu_tensor *latent;
    h3_gpu_tensor *post;
    h3_gpu_tensor *patch_hidden;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *ff1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *projected;
    /* CUDA 3xTF32 fusions: TF32 high/low parts of the normalized and SwiGLU
     * activations (NULL in the FP32 reference mode). */
    h3_gpu_tensor *norm_parts;
    h3_gpu_tensor *act_parts;
    h3_gpu_tensor *zero;           /* resident suffix token */
    h3_gpu_tensor *projected_next; /* second readback slot of the tile pipeline */
    uint32_t patches;
    uint32_t sequence;
    int latent_h;
    int latent_w;
    int latent_t;
    int output_frames;
} vae_context;

#ifdef H3_CUDA
enum { PIPELINE_SLOTS = 2 };
#endif

struct h3_video_vae_decoder {
    vae_context vae;
    tile_axis y_axis;
    tile_axis x_axis;
    int latent_h;
    int latent_w;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
#ifdef H3_CUDA
    /* Tile pipeline: page-locked latent inputs and output patches, and the
     * events that mark each slot's readback as complete. */
    float *pinned_latent[PIPELINE_SLOTS];
    float *pinned_rows[PIPELINE_SLOTS];
    size_t pinned_latent_count;
    size_t pinned_rows_count;
    h3_gpu_event *readback[PIPELINE_SLOTS];
#endif
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(vae_context *vae, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(vae->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *load_f32(vae_context *vae, const char *name, int ndim,
                               const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(vae->weights, vae->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(vae_context *vae, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(vae, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f2(vae_context *vae, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load_f32(vae, name, 2, shape, error, error_size);
}

static void free_block(vae_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
}

static void cleanup(vae_context *vae) {
    if (!vae) return;
    for (int index = 0; index < LAYERS; index++) free_block(&vae->blocks[index]);
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers); free_tensor(&vae->norm_out_w);
    free_tensor(&vae->norm_out_b); free_tensor(&vae->proj_w);
    free_tensor(&vae->proj_b); free_tensor(&vae->latent); free_tensor(&vae->post);
    free_tensor(&vae->patch_hidden); free_tensor(&vae->hidden);
    free_tensor(&vae->norm); free_tensor(&vae->qkv); free_tensor(&vae->query);
    free_tensor(&vae->key); free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->rope_cos);
    free_tensor(&vae->rope_sin); free_tensor(&vae->projected);
    free_tensor(&vae->norm_parts); free_tensor(&vae->act_parts);
    free_tensor(&vae->zero); free_tensor(&vae->projected_next);
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

#ifdef H3_CUDA
/* Video VAE precision on CUDA. FP16 (default): FP16 tensor-core products and
 * attention with FP32 accumulation, FP32 norms and residual stream, the
 * diffusers mixed-precision decode. 3XTF32: products about as accurate as
 * FP32. FP32: the reference path that matches PyTorch's FP32 decode.
 * H3_VAE_PRECISION=fp16|3xtf32|fp32; H3_VAE_3XTF32=0 still selects FP32. */
enum { VAE_FP32, VAE_3XTF32, VAE_FP16 };
static int vae_precision(void) {
    const char *mode = getenv("H3_VAE_PRECISION");
    if (mode && *mode) {
        if (!strcmp(mode, "fp32")) return VAE_FP32;
        if (!strcmp(mode, "3xtf32")) return VAE_3XTF32;
        return VAE_FP16;
    }
    const char *x3 = getenv("H3_VAE_3XTF32");
    return x3 && !strcmp(x3, "0") ? VAE_FP32 : VAE_FP16;
}
static int use_3xtf32(void) { return vae_precision() == VAE_3XTF32; }
static int use_fp16(void) { return vae_precision() == VAE_FP16; }

/* Replace an FP32 weight with its FP16 rounding. */
static int weight_to_f16(vae_context *vae, h3_gpu_tensor **weight, char *error,
                         size_t error_size) {
    h3_gpu_tensor *half = h3_gpu_tensor_to_f16(vae->gpu, *weight);
    if (!half) {
        fail(error, error_size, "cannot convert video VAE weight to FP16: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    h3_gpu_tensor_free(*weight);
    *weight = half;
    return 1;
}
#endif

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    F2(qkv_w, "attn.to_qkv.weight", INNER * 3, HIDDEN);
    F1(qkv_b, "attn.to_qkv.bias", INNER * 3);
    F2(out_w, "attn.to_out.weight", HIDDEN, INNER);
    F1(out_b, "attn.to_out.bias", HIDDEN);
    F1(scale1, "scale1", HIDDEN);
    F1(norm2, "norm2.weight", HIDDEN);
    F2(w1, "ff.w1.weight", FFN * 2, HIDDEN);
    F1(w1_b, "ff.w1.bias", FFN * 2);
    F2(w2, "ff.w2.weight", HIDDEN, FFN);
    F1(w2_b, "ff.w2.bias", HIDDEN);
    F1(scale2, "scale2", HIDDEN);
#undef F1
#undef F2
#ifdef H3_CUDA
    if (use_fp16()) {
        h3_gpu_tensor **half[] = {&block->qkv_w, &block->out_w, &block->w1, &block->w2};
        for (size_t item = 0; item < sizeof(half) / sizeof(half[0]); item++)
            if (!weight_to_f16(vae, half[item], error, error_size)) return 0;
        return 1;
    }
    /* The four block projections run as 3xTF32 tensor-core products instead
     * of FP32 SIMT GEMMs: 2.5x faster, decoded relative error about 1e-4
     * against the FP32 reference, and twice the memory for these weights.
     * H3_VAE_PRECISION=3xtf32 selects it; fp32 is the PyTorch-exact path. */
    /* The same switch selects the fused 3xTF32 attention kernel instead of
     * FP32 SIMT GEMMs + softmax over packed heads. */
    block->tf32_attention = use_3xtf32();
    if (block->tf32_attention) {
        h3_gpu_tensor *split[] = {block->qkv_w, block->out_w, block->w1, block->w2};
        for (size_t item = 0; item < sizeof(split) / sizeof(split[0]); item++)
            if (!h3_gpu_tensor_split_tf32(vae->gpu, split[item])) {
                fail(error, error_size, "cannot split video VAE weight for 3xTF32: %s",
                     h3_gpu_error(vae->gpu));
                return 0;
            }
    }
#endif
    return 1;
}

static int load_input_weights(vae_context *vae, char *error,
                              size_t error_size) {
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1};
    vae->post_w = load_f32(vae, "post_quant_conv.weight", 5, post_shape,
                           error, error_size);
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    vae->embed_w = f2(vae, "decoder.x_embedder.weight", HIDDEN,
                      LATENT_CHANNELS, error, error_size);
    vae->embed_b = f1(vae, "decoder.x_embedder.bias", HIDDEN,
                      error, error_size);
    uint64_t register_shape[] = {1, REGISTERS, HIDDEN};
    vae->registers = load_f32(vae, "decoder.register_tokens", 3,
                              register_shape, error, error_size);
    return vae->post_w && vae->post_b && vae->embed_w && vae->embed_b &&
           vae->registers;
}

static int load_output_weights(vae_context *vae, char *error,
                               size_t error_size) {
    vae->norm_out_w = f1(vae, "decoder.norm_out.weight", HIDDEN,
                          error, error_size);
    vae->norm_out_b = f1(vae, "decoder.norm_out.bias", HIDDEN,
                          error, error_size);
    vae->proj_w = f2(vae, "decoder.proj_out.weight", OUTPUT_PATCH, HIDDEN,
                      error, error_size);
    vae->proj_b = f1(vae, "decoder.proj_out.bias", OUTPUT_PATCH,
                      error, error_size);
    return vae->norm_out_w && vae->norm_out_b && vae->proj_w && vae->proj_b;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open video VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static void fill_input_rows(const vae_context *vae, const float *input,
                            const float *mean, const float *deviation,
                            float *rows) {
    size_t row = 0;
    for (int time = 0; time < CHUNK_LATENT_TIME; time++) {
        int source_t = time < vae->latent_t ? time : vae->latent_t - 1;
        for (int height = 0; height < vae->latent_h; height++)
            for (int width = 0; width < vae->latent_w; width++)
                for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * (size_t)vae->latent_t +
                        (size_t)source_t) *
                        (size_t)vae->latent_h + (size_t)height) *
                        (size_t)vae->latent_w + (size_t)width;
                    rows[row++] = input[source] * deviation[channel] +
                                  mean[channel];
                }
    }
}

static int prepare_input(vae_context *vae, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t patch_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows = malloc(patch_elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory padding video latent");
        return 0;
    }
    fill_input_rows(vae, input, mean, deviation, rows);
    vae->latent = h3_gpu_tensor_from_f32(vae->gpu, rows, patch_elements);
    free(rows);
    return vae->latent != NULL;
}

static int prepare_rope(vae_context *vae, char *error, size_t error_size) {
    size_t count = (size_t)vae->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines); free(sines);
        fail(error, error_size, "out of memory allocating video VAE RoPE");
        return 0;
    }
    uint32_t row = 0;
    for (int t = 0; t < CHUNK_LATENT_TIME; t++)
        for (int h = 0; h < vae->latent_h; h++)
            for (int w = 0; w < vae->latent_w; w++, row++) {
                float axes[] = {
                    2.0f * (((float)t + 0.5f) /
                            (float)CHUNK_LATENT_TIME) - 1.0f,
                    2.0f * (((float)h + 0.5f) / (float)vae->latent_h) - 1.0f,
                    2.0f * (((float)w + 0.5f) / (float)vae->latent_w) - 1.0f
                };
                for (uint32_t axis = 0; axis < 3; axis++)
                    for (uint32_t frequency = 0; frequency < 8; frequency++) {
                        float inverse = 1.0f /
                            powf(100.0f, (float)frequency * 0.125f);
                        float angle = 2.0f * 3.14159265358979323846f *
                                      axes[axis] * inverse;
                        size_t offset = (size_t)row * ROPE_HALF +
                                        axis * 8 + frequency;
                        cosines[offset] = cosf(angle);
                        sines[offset] = sinf(angle);
                    }
            }
    while (row < vae->sequence) {
        for (uint32_t index = 0; index < ROPE_HALF; index++) {
            cosines[(size_t)row * ROPE_HALF + index] = 1.0f;
            sines[(size_t)row * ROPE_HALF + index] = 0.0f;
        }
        row++;
    }
    vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
    vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    free(cosines); free(sines);
    if (!vae->rope_cos || !vae->rope_sin) {
        fail(error, error_size, "cannot allocate video VAE RoPE: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static int allocate_activations(vae_context *vae, char *error,
                                size_t error_size) {
    size_t patches = vae->patches, sequence = vae->sequence;
    /* In FP16 mode the product inputs (normed rows, Q/K/V, attention output,
     * SwiGLU output) are FP16; everything else stays FP32. */
    int half = 0;
#ifdef H3_CUDA
    half = use_fp16();
#define F16(field, elements) (vae->field = half                                 \
        ? h3_gpu_tensor_new_f16(vae->gpu, (elements))                          \
        : h3_gpu_tensor_new_f32(vae->gpu, (elements)))
#else
#define F16(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
#endif
    (void)half;
#define F32(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(post, patches * LATENT_CHANNELS),
        F32(patch_hidden, patches * HIDDEN),
        F32(hidden, sequence * HIDDEN),
        F16(norm, sequence * HIDDEN),
        F32(qkv, sequence * INNER * 3),
        F16(query, sequence * INNER), F16(key, sequence * INNER),
        F16(value, sequence * INNER), F16(heads, sequence * INNER),
        F32(branch, sequence * HIDDEN), F32(ff1, sequence * FFN * 2),
        F16(activated, sequence * FFN),
        F32(projected, sequence * OUTPUT_PATCH)
    };
#undef F32
#undef F16
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate video VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
#ifdef H3_CUDA
    if (use_3xtf32()) {
        vae->norm_parts = h3_gpu_tensor_new_f32(vae->gpu, 2 * sequence * HIDDEN);
        vae->act_parts = h3_gpu_tensor_new_f32(vae->gpu, 2 * sequence * FFN);
        if (!vae->norm_parts || !vae->act_parts) {
            fail(error, error_size, "cannot allocate video VAE TF32 parts: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    }
#endif
    return 1;
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
#ifdef H3_CUDA
    /* FP16: each product reads FP16 inputs and returns FP32; the bias is
     * added in FP32 where the product is consumed. */
    if (h3_gpu_tensor_dtype(weight->qkv_w) == H3_GPU_F16) {
        OP(h3_gpu_rms_norm_f16_out(vae->gpu, vae->norm, vae->hidden,
            weight->norm1, rows, HIDDEN, 1e-5f), "video VAE attention norm");
        OP(h3_gpu_linear_f16_f32_out(vae->gpu, vae->qkv, vae->norm,
            weight->qkv_w, rows, HIDDEN, INNER * 3), "video VAE QKV");
        OP(h3_gpu_video_qkv_rope_bias_f16(vae->gpu, vae->query, vae->key,
            vae->value, vae->qkv, weight->qkv_b, vae->rope_cos, vae->rope_sin,
            rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
        OP(h3_gpu_sdpa_f16_d64(vae->gpu, vae->heads, vae->query, vae->key,
            vae->value, rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "video VAE attention");
        OP(h3_gpu_linear_f16_f32_out(vae->gpu, vae->branch, vae->heads,
            weight->out_w, rows, INNER, HIDDEN), "video VAE attention output");
        OP(h3_gpu_scale_add_bias_f32(vae->gpu, vae->hidden, vae->hidden,
            vae->branch, weight->out_b, weight->scale1, rows, HIDDEN),
           "video VAE attention residual");
        OP(h3_gpu_rms_norm_f16_out(vae->gpu, vae->norm, vae->hidden,
            weight->norm2, rows, HIDDEN, 1e-5f), "video VAE MLP norm");
        OP(h3_gpu_linear_f16_f32_out(vae->gpu, vae->ff1, vae->norm,
            weight->w1, rows, HIDDEN, FFN * 2), "video VAE MLP input");
        OP(h3_gpu_swiglu_bias_f16_out(vae->gpu, vae->activated, vae->ff1,
            weight->w1_b, rows, FFN), "video VAE SwiGLU");
        OP(h3_gpu_linear_f16_f32_out(vae->gpu, vae->branch, vae->activated,
            weight->w2, rows, FFN, HIDDEN), "video VAE MLP output");
        OP(h3_gpu_scale_add_bias_f32(vae->gpu, vae->hidden, vae->hidden,
            vae->branch, weight->w2_b, weight->scale2, rows, HIDDEN),
           "video VAE MLP residual");
        return 1;
    }
    /* 3xTF32: the norms and SwiGLU write TF32 parts for the next product, and
     * each bias is added where its product is consumed. Same values as the
     * unfused path below with split weights. */
    if (vae->norm_parts && vae->act_parts &&
        h3_gpu_tensor_is_split_tf32(weight->qkv_w) &&
        h3_gpu_tensor_is_split_tf32(weight->w1) &&
        h3_gpu_tensor_is_split_tf32(weight->w2)) {
        OP(h3_gpu_rms_norm_split_tf32(vae->gpu, vae->norm_parts, vae->hidden,
            weight->norm1, rows, HIDDEN, 1e-5f), "video VAE attention norm");
        OP(h3_gpu_linear_split_input_f32(vae->gpu, vae->qkv, vae->norm_parts,
            weight->qkv_w, rows, HIDDEN, INNER * 3), "video VAE QKV");
        OP(h3_gpu_video_qkv_rope_bias_f32(vae->gpu, vae->query, vae->key,
            vae->value, vae->qkv, weight->qkv_b, vae->rope_cos, vae->rope_sin,
            rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
        if (weight->tf32_attention)
            OP(h3_gpu_sdpa_3xtf32_f32(vae->gpu, vae->heads, vae->query,
                vae->key, vae->value, rows, HEADS, HEAD_DIM,
                1.0f / sqrtf((float)HEAD_DIM)), "video VAE attention");
        else
            OP(h3_gpu_sdpa_f32(vae->gpu, vae->heads, vae->query, vae->key,
                vae->value, rows, HEADS, HEAD_DIM,
                1.0f / sqrtf((float)HEAD_DIM)), "video VAE attention");
        OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->heads, weight->out_w,
            NULL, rows, INNER, HIDDEN), "video VAE attention output");
        OP(h3_gpu_scale_add_bias_f32(vae->gpu, vae->hidden, vae->hidden,
            vae->branch, weight->out_b, weight->scale1, rows, HIDDEN),
           "video VAE attention residual");
        OP(h3_gpu_rms_norm_split_tf32(vae->gpu, vae->norm_parts, vae->hidden,
            weight->norm2, rows, HIDDEN, 1e-5f), "video VAE MLP norm");
        OP(h3_gpu_linear_split_input_f32(vae->gpu, vae->ff1, vae->norm_parts,
            weight->w1, rows, HIDDEN, FFN * 2), "video VAE MLP input");
        OP(h3_gpu_swiglu_bias_split_tf32(vae->gpu, vae->act_parts, vae->ff1,
            weight->w1_b, rows, FFN), "video VAE SwiGLU");
        OP(h3_gpu_linear_split_input_f32(vae->gpu, vae->branch, vae->act_parts,
            weight->w2, rows, FFN, HIDDEN), "video VAE MLP output");
        OP(h3_gpu_scale_add_bias_f32(vae->gpu, vae->hidden, vae->hidden,
            vae->branch, weight->w2_b, weight->scale2, rows, HIDDEN),
           "video VAE MLP residual");
        return 1;
    }
#endif
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "video VAE attention norm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->qkv, vae->norm, weight->qkv_w,
        weight->qkv_b, rows, HIDDEN, INNER * 3), "video VAE QKV");
    OP(h3_gpu_video_qkv_rope_f32(vae->gpu, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
#ifdef H3_CUDA
    if (weight->tf32_attention)
        OP(h3_gpu_sdpa_3xtf32_f32(vae->gpu, vae->heads, vae->query, vae->key,
            vae->value, rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "video VAE attention");
    else
#endif
    OP(h3_gpu_sdpa_f32(vae->gpu, vae->heads, vae->query, vae->key, vae->value,
        rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "video VAE attention");
    OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->heads, weight->out_w,
        weight->out_b, rows, INNER, HIDDEN), "video VAE attention output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "video VAE attention residual");
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm2,
        rows, HIDDEN, 1e-5f), "video VAE MLP norm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->ff1, vae->norm, weight->w1,
        weight->w1_b, rows, HIDDEN, FFN * 2), "video VAE MLP input");
    OP(h3_gpu_swiglu_f32(vae->gpu, vae->activated, vae->ff1, rows, FFN),
       "video VAE SwiGLU");
    OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->activated, weight->w2,
        weight->w2_b, rows, FFN, HIDDEN), "video VAE MLP output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "video VAE MLP residual");
#undef OP
    return 1;
}

/* The output LayerNorm is FP32 in every mode; in FP16 mode `norm` is an FP16
 * buffer, so the free FP32 `branch` buffer holds it instead. */
static h3_gpu_tensor *output_norm(vae_context *vae) {
    return h3_gpu_tensor_dtype(vae->norm) == H3_GPU_F32 ? vae->norm : vae->branch;
}

static int run_decoder(vae_context *vae, h3_video_vae_progress progress,
                       void *progress_opaque, char *error,
                       size_t error_size) {
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin video VAE decoder");
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack video VAE suffix");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE input projection");
    h3_gpu_tensor_free(zero);
    zero = NULL;
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers);

    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_begin(vae->gpu), "begin video VAE transformer block");
        if (!run_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_submit(vae->gpu), "submit video VAE transformer block");
        free_block(&vae->blocks[index]);
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    if (!load_output_weights(vae, error, error_size)) return 0;
    OP(h3_gpu_begin(vae->gpu), "begin video VAE output projection");
    OP(h3_gpu_layer_norm_f32(vae->gpu, output_norm(vae), vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, output_norm(vae), vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "video VAE output projection");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE decoder");
    free_tensor(&vae->norm_out_w); free_tensor(&vae->norm_out_b);
    free_tensor(&vae->proj_w); free_tensor(&vae->proj_b);
#undef OP
    return 1;
}

static int load_resident_weights(vae_context *vae,
                                 h3_video_vae_progress progress,
                                 void *progress_opaque, char *error,
                                 size_t error_size) {
    if (!load_input_weights(vae, error, error_size)) return 0;
    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    return load_output_weights(vae, error, error_size);
}

/* Tiled decoding keeps every VAE weight resident for the duration of the phase.
 * This uses about 9 GiB after the DiT has been retired, avoids rereading 9 GiB
 * per spatial tile, and permits one command-buffer chain per tile. */
static int run_resident_tile(vae_context *vae, h3_gpu_tensor *projected,
                             int synchronous, char *error,
                             size_t error_size) {
    h3_gpu_tensor *zero = vae->zero;
    if (!zero) {
        float zeros[HIDDEN];
        memset(zeros, 0, sizeof(zeros));
        zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    }
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define RELEASE_ZERO() do { if (zero != vae->zero) h3_gpu_tensor_free(zero); } while (0)
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        RELEASE_ZERO(); return 0;                                               \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin tiled video VAE decoder");
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "tiled video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "tiled video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack tiled video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack tiled video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack tiled video VAE suffix");
    for (int index = 0; index < LAYERS; index++)
        if (!run_block(vae, index, error, error_size)) {
            RELEASE_ZERO();
            return 0;
        }
    OP(h3_gpu_layer_norm_f32(vae->gpu, output_norm(vae), vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "tiled video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, projected, output_norm(vae), vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "tiled video VAE output projection");
    if (synchronous)
        OP(h3_gpu_submit(vae->gpu), "submit tiled video VAE decoder");
    RELEASE_ZERO();
#undef OP
#undef RELEASE_ZERO
    return 1;
}

/* Rearrange output patches (sequence x OUTPUT_PATCH) into RGB frames.
 * Touches no GPU state, so the CUDA tile pipeline runs it on a host thread. */
static int unpack_rows(const vae_context *vae, const float *rows,
                       int first_frame, int frame_count,
                       h3_video_frames *output, char *error,
                       size_t error_size) {
    if (first_frame < 0 || frame_count < 1 ||
        first_frame > vae->output_frames - frame_count) {
        fail(error, error_size, "invalid video VAE output frame range");
        return 0;
    }
    int pixel_h = vae->latent_h * 16;
    int pixel_w = vae->latent_w * 16;
    size_t rgb_count = (size_t)frame_count * (size_t)pixel_h *
                       (size_t)pixel_w * 3;
    float *rgb = malloc(rgb_count * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory unpacking video VAE frames");
        return 0;
    }
#ifndef H3_CUDA
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
#endif
    for (int output_frame = 0; output_frame < frame_count; output_frame++) {
        int frame = first_frame + output_frame;
        int decoded_t = frame + FRAME_OFFSET;
        if (vae->output_frames == FIRST_CHUNK_FRAMES && frame >= 17)
            decoded_t += 3;
        int patch_t = decoded_t / 4;
        int within_t = decoded_t % 4;
        for (int y = 0; y < pixel_h; y++) {
            int patch_y = y / 16, within_y = y % 16;
            for (int x = 0; x < pixel_w; x++) {
                int patch_x = x / 16, within_x = x % 16;
                size_t patch = ((size_t)patch_t * (size_t)vae->latent_h +
                    (size_t)patch_y) * (size_t)vae->latent_w +
                    (size_t)patch_x;
                for (int channel = 0; channel < 3; channel++) {
                    size_t component = ((((size_t)channel * 4 +
                        (size_t)within_t) * 16 + (size_t)within_y) * 16 +
                        (size_t)within_x);
#ifdef H3_CUDA
                    float value = rows[patch * OUTPUT_PATCH + component];
#else
                    float value = rows[patch * OUTPUT_PATCH + component] *
                                  deviation[channel] + mean[channel];
                    if (value < 0.0f) value = 0.0f;
                    if (value > 1.0f) value = 1.0f;
#endif
                    size_t destination = (((size_t)output_frame *
                        (size_t)pixel_h +
                        (size_t)y) * (size_t)pixel_w + (size_t)x) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
            }
        }
    }
    output->frames = frame_count;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = rgb;
    return 1;
}

static int unpack_frame_range(vae_context *vae, int first_frame,
                              int frame_count, h3_video_frames *output,
                              char *error, size_t error_size) {
    size_t projected_count = (size_t)vae->sequence * OUTPUT_PATCH;
    float *rows = malloc(projected_count * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory unpacking video VAE frames");
        return 0;
    }
    if (!h3_gpu_tensor_read_f32(vae->projected, rows, projected_count)) {
        free(rows);
        fail(error, error_size, "cannot read video VAE output patches");
        return 0;
    }
    int ok = unpack_rows(vae, rows, first_frame, frame_count, output,
                         error, error_size);
    free(rows);
    return ok && h3_gpu_get_stats(vae->gpu, &output->gpu_stats);
}

static int unpack_frames(vae_context *vae, h3_video_frames *output,
                         char *error, size_t error_size) {
    return unpack_frame_range(vae, 0, vae->output_frames, output,
                              error, error_size);
}

#ifdef H3_CUDA
/* rgb holds whole pixels (count is a multiple of 3). */
static void finalize_pixel_range(float *rgb, size_t count) {
    /* The reference blends raw decoder values before pixel normalization
     * and clipping. Clipping individual tiles changes overlap pixels. */
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    for (size_t index = 0; index < count; ++index) {
        float value = rgb[index] * deviation[index % 3] + mean[index % 3];
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        rgb[index] = value;
    }
}

/* Same conversion as h3.c's RGB24 output. */
static void pixels_to_rgb8(const float *rgb, uint8_t *output, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float scaled = rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        output[index] = (uint8_t)lrintf(scaled);
    }
}

static void finalize_video_pixels(h3_video_frames *output) {
    finalize_pixel_range(output->rgb,
        (size_t)output->frames * output->height * output->width * 3);
}
#endif

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_count_for_extent(int extent, int tile_pixels) {
    if (extent <= tile_pixels) return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    return count;
}

static int configured_tile_pixels(int pixel_height, int pixel_width) {
#ifdef H3_CUDA
    /* The released reference enables fixed 256-pixel tiles. Adaptive tile
     * sizes change each tile's attention context and therefore its output. */
    (void)pixel_height;
    (void)pixel_width;
    return TILE_PIXELS;
#else
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    if (value && *value) {
        char *end = NULL;
        long pixels = strtol(value, &end, 10);
        if (end && !*end && pixels >= TILE_PIXELS && pixels <= 512 &&
            pixels % SPATIAL_RATIO == 0) return (int)pixels;
    }
    int best = TILE_PIXELS;
    uint64_t best_score = UINT64_MAX;
    for (int pixels = TILE_PIXELS; pixels <= 320;
         pixels += SPATIAL_RATIO) {
        uint64_t tiles = (uint64_t)tile_count_for_extent(pixel_height, pixels) *
            (uint64_t)tile_count_for_extent(pixel_width, pixels);
        /* The resident VAE is dominated by linears and activation traffic;
         * measured tile cost follows area more closely than cubic sequence
         * growth at these shapes. Attention is still included in each tile. */
        uint64_t score = tiles * (uint64_t)pixels * (uint64_t)pixels;
        if (score < best_score) {
            best = pixels;
            best_score = score;
        }
    }
    return best;
#endif
}

static int tile_axis_build(int extent, int tile_pixels, tile_axis *axis,
                           char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < 1 || extent % SPATIAL_RATIO) {
        fail(error, error_size, "video VAE tile extent must be a multiple of %d",
             SPATIAL_RATIO);
        return 0;
    }
    if (extent <= tile_pixels) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing VAE tile plan");
            return 0;
        }
        return 1;
    }
    int count = tile_count_for_extent(extent, tile_pixels);
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing VAE tile plan");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + tile_pixels -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = tile_pixels;
    return 1;
}

static float *extract_latent_tile(const float *latent, int full_t,
                                  int full_h, int full_w, int start_t,
                                  int start_y, int start_x, int tile_t,
                                  int tile_h, int tile_w, char *error,
                                  size_t error_size) {
    size_t count = (size_t)LATENT_CHANNELS * (size_t)tile_t *
                   (size_t)tile_h * (size_t)tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting video VAE tile");
        return NULL;
    }
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int time = 0; time < tile_t; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * (size_t)full_t +
                    (size_t)(start_t + time)) * (size_t)full_h +
                    (size_t)(start_y + y)) * (size_t)full_w +
                    (size_t)start_x;
                size_t destination = (((size_t)channel * (size_t)tile_t +
                    (size_t)time) * (size_t)tile_h + (size_t)y) *
                    (size_t)tile_w;
                memcpy(tile + destination, latent + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

/* Write the region of tile `index` that it owns, blended with the tiles
 * above and to the left. It reads only those three tiles, so tiles can be
 * stitched in row-major order as they arrive. */
static void stitch_tile(float *const *tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int frame_count, int index,
                        float *rgb) {
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    int tile_y = index / x_axis->count;
    int tile_x = index % x_axis->count;
    const float *current = tiles[index];
    const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
    const float *left = tile_x ? tiles[index - 1] : NULL;
    int overlap_y = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
    int overlap_x = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
    int keep_h = y_axis->length -
        (tile_y + 1 < y_axis->count ? y_axis->overlaps[tile_y] : 0);
    int keep_w = x_axis->length -
        (tile_x + 1 < x_axis->count ? x_axis->overlaps[tile_x] : 0);
    for (int frame = 0; frame < frame_count; frame++)
        for (int y = 0; y < keep_h; y++)
            for (int x = 0; x < keep_w; x++)
                for (int channel = 0; channel < 3; channel++) {
                    size_t local = (((size_t)frame *
                        (size_t)y_axis->length + (size_t)y) *
                        (size_t)x_axis->length + (size_t)x) * 3 +
                        (size_t)channel;
                    float value = current[local];
                    if (above && y < overlap_y) {
                        size_t top = (((size_t)frame *
                            (size_t)y_axis->length +
                            (size_t)(y_axis->length - overlap_y + y)) *
                            (size_t)x_axis->length + (size_t)x) * 3 +
                            (size_t)channel;
                        float alpha = (float)y / (float)overlap_y;
                        value = above[top] * (1.0f - alpha) +
                                value * alpha;
                    }
                    if (left && x < overlap_x) {
                        size_t prior = (((size_t)frame *
                            (size_t)y_axis->length + (size_t)y) *
                            (size_t)x_axis->length +
                            (size_t)(x_axis->length - overlap_x + x)) *
                            3 + (size_t)channel;
                        float alpha = (float)x / (float)overlap_x;
                        value = left[prior] * (1.0f - alpha) +
                                value * alpha;
                    }
                    size_t destination = (((size_t)frame *
                        (size_t)full_h +
                        (size_t)(y_axis->starts[tile_y] + y)) *
                        (size_t)full_w +
                        (size_t)(x_axis->starts[tile_x] + x)) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
}

static int stitch_tiles(float **tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int frame_count,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    size_t count = (size_t)frame_count * (size_t)full_h *
                   (size_t)full_w * 3;
    float *rgb = malloc(count * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory stitching video VAE tiles");
        return 0;
    }
    for (int index = 0; index < y_axis->count * x_axis->count; index++)
        stitch_tile(tiles, y_axis, x_axis, frame_count, index, rgb);
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    return 1;
}

#ifdef H3_CUDA
/* Tile pipeline. The main thread queues tile k+1 on the GPU while a host
 * thread rearranges, stitches, blends and finalizes tile k. Each tile's
 * output patches go to a page-locked buffer through an asynchronous copy on a
 * separate stream. Two slots alternate; a slot is reused only after the host
 * thread has released the tile that used it. The GPU and host operations are
 * the same as in the serial path, in the same order per value, so the output
 * is bit-identical. */
typedef struct vae_pipeline vae_pipeline;
typedef int (*vae_chunk_sink)(vae_pipeline *pipeline, int chunk,
                              h3_video_frames *stitched, char *error,
                              size_t error_size);

struct vae_pipeline {
    h3_video_vae_decoder *decoder;
    const float *latent;
    int latent_time;
    int first_chunk;
    int chunk_count;
    int selected_frame;  /* -1: all FIRST_CHUNK_FRAMES frames per tile */
    int frame_count;
    int tile_count;
    int items;
    vae_chunk_sink sink;  /* runs on the host thread; owns `stitched` */
    void *sink_state;
    float **tiles;
    h3_video_frames stitched;  /* chunk being stitched (host thread) */
    pthread_mutex_t lock;
    pthread_cond_t changed;
    int posted;    /* tiles queued on the GPU */
    int released;  /* tiles whose slot the host thread has finished reading */
    int stop;
    int failed;
    char error[512];
};

static void pipeline_set(vae_pipeline *pipeline, int *field, int value,
                         const char *error) {
    pthread_mutex_lock(&pipeline->lock);
    *field = value;
    if (error && !pipeline->failed) {
        pipeline->failed = 1;
        snprintf(pipeline->error, sizeof(pipeline->error), "%s", error);
    }
    pthread_cond_broadcast(&pipeline->changed);
    pthread_mutex_unlock(&pipeline->lock);
}

static void *pipeline_worker(void *opaque) {
    vae_pipeline *pipeline = opaque;
    h3_video_vae_decoder *decoder = pipeline->decoder;
    char error[512] = "";
    for (int item = 0; item < pipeline->items; item++) {
        pthread_mutex_lock(&pipeline->lock);
        while (pipeline->posted <= item && !pipeline->stop)
            pthread_cond_wait(&pipeline->changed, &pipeline->lock);
        int queued = pipeline->posted > item;
        pthread_mutex_unlock(&pipeline->lock);
        if (!queued) break;
        int slot = item % PIPELINE_SLOTS;
        h3_video_frames tile;
        memset(&tile, 0, sizeof(tile));
        int ok = h3_gpu_event_wait(decoder->readback[slot], error, sizeof(error)) &&
            unpack_rows(&decoder->vae, decoder->pinned_rows[slot],
                        pipeline->selected_frame >= 0 ? pipeline->selected_frame : 0,
                        pipeline->frame_count, &tile, error, sizeof(error));
        pipeline_set(pipeline, &pipeline->released, item + 1, ok ? NULL : error);
        if (!ok) break;
        int index = item % pipeline->tile_count;
        pipeline->tiles[index] = tile.rgb;
        const tile_axis *y_axis = &decoder->y_axis, *x_axis = &decoder->x_axis;
        int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
        int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
        if (!index) {
            memset(&pipeline->stitched, 0, sizeof(pipeline->stitched));
            pipeline->stitched.rgb = malloc((size_t)pipeline->frame_count *
                (size_t)full_h * (size_t)full_w * 3 * sizeof(float));
            if (!pipeline->stitched.rgb) {
                pipeline_set(pipeline, &pipeline->stop, 1,
                             "out of memory stitching video VAE tiles");
                break;
            }
        }
        /* Stitch each tile as it arrives, so only the last tile's region
         * is left when the GPU finishes the chunk. */
        stitch_tile(pipeline->tiles, y_axis, x_axis, pipeline->frame_count,
                    index, pipeline->stitched.rgb);
        if (index + 1 < pipeline->tile_count) continue;
        h3_video_frames stitched = pipeline->stitched;
        memset(&pipeline->stitched, 0, sizeof(pipeline->stitched));
        stitched.frames = pipeline->frame_count;
        stitched.height = full_h;
        stitched.width = full_w;
        for (int tile_index = 0; tile_index < pipeline->tile_count; tile_index++) {
            free(pipeline->tiles[tile_index]);
            pipeline->tiles[tile_index] = NULL;
        }
        ok = pipeline->sink(pipeline,
                            pipeline->first_chunk + item / pipeline->tile_count,
                            &stitched, error, sizeof(error));
        if (!ok) {
            pipeline_set(pipeline, &pipeline->stop, 1, error);
            break;
        }
    }
    return NULL;
}

/* Resident buffers for the pipeline; reallocated only when the tile geometry
 * changes. */
static int pipeline_prepare(h3_video_vae_decoder *decoder, char *error,
                            size_t error_size) {
    vae_context *vae = &decoder->vae;
    size_t latent_count = (size_t)vae->patches * LATENT_CHANNELS;
    size_t rows_count = (size_t)vae->sequence * OUTPUT_PATCH;
    if (decoder->pinned_latent_count != latent_count ||
        decoder->pinned_rows_count != rows_count) {
        for (int slot = 0; slot < PIPELINE_SLOTS; slot++) {
            h3_gpu_host_free(decoder->pinned_latent[slot]);
            h3_gpu_host_free(decoder->pinned_rows[slot]);
            decoder->pinned_latent[slot] = NULL;
            decoder->pinned_rows[slot] = NULL;
        }
        decoder->pinned_latent_count = decoder->pinned_rows_count = 0;
        for (int slot = 0; slot < PIPELINE_SLOTS; slot++) {
            decoder->pinned_latent[slot] =
                h3_gpu_host_alloc(latent_count * sizeof(float));
            decoder->pinned_rows[slot] =
                h3_gpu_host_alloc(rows_count * sizeof(float));
            if (!decoder->pinned_latent[slot] || !decoder->pinned_rows[slot]) {
                fail(error, error_size, "cannot allocate page-locked video VAE buffers");
                return 0;
            }
        }
        decoder->pinned_latent_count = latent_count;
        decoder->pinned_rows_count = rows_count;
    }
    for (int slot = 0; slot < PIPELINE_SLOTS; slot++)
        if (!decoder->readback[slot] &&
            !(decoder->readback[slot] = h3_gpu_event_new(vae->gpu))) {
            fail(error, error_size, "cannot create video VAE readback event: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    if (vae->latent && h3_gpu_tensor_elements(vae->latent) != latent_count)
        free_tensor(&vae->latent);
    if (!vae->latent) vae->latent = h3_gpu_tensor_new_f32(vae->gpu, latent_count);
    if (!vae->projected_next)
        vae->projected_next = h3_gpu_tensor_new_f32(vae->gpu, rows_count);
    if (!vae->zero) {
        float zeros[HIDDEN];
        memset(zeros, 0, sizeof(zeros));
        vae->zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    }
    if (!vae->latent || !vae->projected_next || !vae->zero) {
        fail(error, error_size, "cannot allocate video VAE pipeline tensors: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static int pipeline_run(vae_pipeline *pipeline, char *error, size_t error_size) {
    h3_video_vae_decoder *decoder = pipeline->decoder;
    vae_context *vae = &decoder->vae;
    pipeline->tile_count = decoder->y_axis.count * decoder->x_axis.count;
    pipeline->items = pipeline->chunk_count * pipeline->tile_count;
    if (!pipeline_prepare(decoder, error, error_size)) return 0;
    pipeline->tiles = calloc((size_t)pipeline->tile_count, sizeof(*pipeline->tiles));
    if (!pipeline->tiles) {
        fail(error, error_size, "out of memory retaining video VAE tiles");
        return 0;
    }
    pthread_mutex_init(&pipeline->lock, NULL);
    pthread_cond_init(&pipeline->changed, NULL);
    pthread_t worker;
    if (pthread_create(&worker, NULL, pipeline_worker, pipeline)) {
        pthread_cond_destroy(&pipeline->changed);
        pthread_mutex_destroy(&pipeline->lock);
        free(pipeline->tiles);
        fail(error, error_size, "cannot start video VAE host thread");
        return 0;
    }
    h3_gpu_tensor *outputs[PIPELINE_SLOTS] = {vae->projected, vae->projected_next};
    size_t latent_count = decoder->pinned_latent_count;
    size_t rows_count = decoder->pinned_rows_count;
    int ok = gpu_op(vae, h3_gpu_begin(vae->gpu), error, error_size,
                    "begin pipelined video VAE decoder");
    for (int item = 0; item < pipeline->items && ok; item++) {
        int slot = item % PIPELINE_SLOTS;
        pthread_mutex_lock(&pipeline->lock);
        while (pipeline->released < item - PIPELINE_SLOTS + 1 && !pipeline->failed)
            pthread_cond_wait(&pipeline->changed, &pipeline->lock);
        int failed = pipeline->failed;
        pthread_mutex_unlock(&pipeline->lock);
        if (failed) { ok = 0; break; }
        int chunk = pipeline->first_chunk + item / pipeline->tile_count;
        int tile = item % pipeline->tile_count;
        int tile_y = tile / decoder->x_axis.count;
        int tile_x = tile % decoder->x_axis.count;
        float *input = extract_latent_tile(
            pipeline->latent, pipeline->latent_time, decoder->latent_h,
            decoder->latent_w, chunk * 5,
            decoder->y_axis.starts[tile_y] / SPATIAL_RATIO,
            decoder->x_axis.starts[tile_x] / SPATIAL_RATIO,
            CHUNK_LATENT_TIME, vae->latent_h, vae->latent_w, error, error_size);
        if (!input) { ok = 0; break; }
        fill_input_rows(vae, input, decoder->latent_mean, decoder->latent_std,
                        decoder->pinned_latent[slot]);
        free(input);
        ok = gpu_op(vae, h3_gpu_tensor_upload_f32_async(vae->gpu, vae->latent,
                        decoder->pinned_latent[slot], latent_count),
                    error, error_size, "upload video VAE tile latent") &&
            /* The readback of this slot's previous tile must finish before the
             * output projection overwrites the slot. */
            (item < PIPELINE_SLOTS ||
             gpu_op(vae, h3_gpu_stream_wait_event(vae->gpu, decoder->readback[slot]),
                    error, error_size, "order video VAE readback")) &&
            run_resident_tile(vae, outputs[slot], 0, error, error_size) &&
            gpu_op(vae, h3_gpu_tensor_download_f32_async(vae->gpu, outputs[slot],
                        decoder->pinned_rows[slot], rows_count,
                        decoder->readback[slot]),
                   error, error_size, "read video VAE output patches");
        if (ok) pipeline_set(pipeline, &pipeline->posted, item + 1, NULL);
    }
    /* Wait for the compute stream (and close the timing interval) even after
     * a failure, so no queued copy still targets the page-locked buffers. */
    int submitted = h3_gpu_submit(vae->gpu);
    if (ok && !submitted)
        ok = gpu_op(vae, 0, error, error_size, "submit pipelined video VAE decoder");
    if (!ok) pipeline_set(pipeline, &pipeline->stop, 1, NULL);
    pthread_join(worker, NULL);
    for (int slot = 0; slot < PIPELINE_SLOTS; slot++)
        h3_gpu_event_wait(decoder->readback[slot], NULL, 0);
    if (ok && pipeline->failed) {
        fail(error, error_size, "%s", pipeline->error);
        ok = 0;
    }
    for (int index = 0; index < pipeline->tile_count; index++)
        free(pipeline->tiles[index]);
    free(pipeline->tiles);
    h3_video_frames_free(&pipeline->stitched);
    pthread_cond_destroy(&pipeline->changed);
    pthread_mutex_destroy(&pipeline->lock);
    return ok;
}

static int single_chunk_sink(vae_pipeline *pipeline, int chunk,
                             h3_video_frames *stitched, char *error,
                             size_t error_size) {
    (void)chunk; (void)error; (void)error_size;
    *(h3_video_frames *)pipeline->sink_state = *stitched;
    return 1;
}

static int decoder_decode_chunk(h3_video_vae_decoder *decoder,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int selected_frame,
                                h3_video_frames *output,
                                char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!decoder || !normalized_latent || !output || latent_time < 7 ||
        chunk < 0 || chunk > (latent_time - CHUNK_LATENT_TIME) / 5 ||
        selected_frame < -1 || selected_frame >= FIRST_CHUNK_FRAMES) {
        fail(error, error_size, "invalid resident video VAE chunk arguments");
        return 0;
    }
    vae_pipeline pipeline;
    memset(&pipeline, 0, sizeof(pipeline));
    pipeline.decoder = decoder;
    pipeline.latent = normalized_latent;
    pipeline.latent_time = latent_time;
    pipeline.first_chunk = chunk;
    pipeline.chunk_count = 1;
    pipeline.selected_frame = selected_frame;
    pipeline.frame_count = selected_frame >= 0 ? 1 : FIRST_CHUNK_FRAMES;
    pipeline.sink = single_chunk_sink;
    pipeline.sink_state = output;
    int ok = pipeline_run(&pipeline, error, error_size);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

/* Temporal blending of consecutive chunks, as in the serial decoder. Frames
 * that are final are normalized and clipped here, on the host thread. */
typedef struct {
    float *final_rgb;
    float *temporal_overlap;
    size_t frame_elements;
    int chunks;
    uint8_t *rgb8;  /* optional RGB24 copy of the final frames */
    h3_video_frames_ready ready;
    void *ready_opaque;
} video_sink_state;

static void emit_final_frames(video_sink_state *state, int first_frame,
                              int frame_count) {
    if (!state->rgb8) return;
    size_t offset = (size_t)first_frame * state->frame_elements;
    pixels_to_rgb8(state->final_rgb + offset, state->rgb8 + offset,
                   (size_t)frame_count * state->frame_elements);
    if (state->ready)
        state->ready(state->ready_opaque, state->rgb8 + offset, first_frame,
                     frame_count);
}

static int video_chunk_sink(vae_pipeline *pipeline, int chunk,
                            h3_video_frames *decoded, char *error,
                            size_t error_size) {
    (void)error; (void)error_size;
    video_sink_state *state = pipeline->sink_state;
    size_t frame_elements = state->frame_elements;
    if (chunk) for (int frame = 0; frame < 5; frame++) {
        float alpha = (float)frame / 5.0f;
        size_t base = (size_t)frame * frame_elements;
        for (size_t index = 0; index < frame_elements; index++)
            decoded->rgb[base + index] = state->temporal_overlap[base + index] *
                (1.0f - alpha) + decoded->rgb[base + index] * alpha;
    }
    float *final_frames = state->final_rgb + (size_t)chunk * 17 * frame_elements;
    memcpy(final_frames, decoded->rgb, 17 * frame_elements * sizeof(float));
    finalize_pixel_range(final_frames, 17 * frame_elements);
    memcpy(state->temporal_overlap, decoded->rgb + 17 * frame_elements,
           5 * frame_elements * sizeof(float));
    h3_video_frames_free(decoded);
    emit_final_frames(state, chunk * 17, 17);
    if (chunk + 1 == state->chunks) {
        float *tail = state->final_rgb + (size_t)state->chunks * 17 * frame_elements;
        memcpy(tail, state->temporal_overlap, 5 * frame_elements * sizeof(float));
        finalize_pixel_range(tail, 5 * frame_elements);
        emit_final_frames(state, state->chunks * 17, 5);
    }
    return 1;
}
#else
static int decoder_decode_chunk(h3_video_vae_decoder *decoder,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int selected_frame,
                                h3_video_frames *output,
                                char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!decoder || !normalized_latent || !output || latent_time < 7 ||
        chunk < 0 || chunk > (latent_time - CHUNK_LATENT_TIME) / 5 ||
        selected_frame < -1 || selected_frame >= FIRST_CHUNK_FRAMES) {
        fail(error, error_size, "invalid resident video VAE chunk arguments");
        return 0;
    }
    int tile_count = decoder->y_axis.count * decoder->x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory retaining video VAE tiles");
        return 0;
    }
    int frame_count = selected_frame >= 0 ? 1 : FIRST_CHUNK_FRAMES;
    int ok = 1;
    for (int tile_y = 0; tile_y < decoder->y_axis.count && ok; tile_y++)
        for (int tile_x = 0; tile_x < decoder->x_axis.count && ok; tile_x++) {
            float *input = extract_latent_tile(
                normalized_latent, latent_time, decoder->latent_h,
                decoder->latent_w, chunk * 5,
                decoder->y_axis.starts[tile_y] / SPATIAL_RATIO,
                decoder->x_axis.starts[tile_x] / SPATIAL_RATIO,
                CHUNK_LATENT_TIME, decoder->vae.latent_h,
                decoder->vae.latent_w, error, error_size);
            if (!input) {
                ok = 0;
                break;
            }
            free_tensor(&decoder->vae.latent);
            ok = prepare_input(&decoder->vae, input,
                               decoder->latent_mean, decoder->latent_std,
                               error, error_size) &&
                 run_resident_tile(&decoder->vae, decoder->vae.projected, 1,
                                   error, error_size);
            free(input);
            if (!ok) break;
            h3_video_frames tile;
            memset(&tile, 0, sizeof(tile));
            ok = selected_frame >= 0 ?
                unpack_frame_range(&decoder->vae, selected_frame, 1, &tile,
                                   error, error_size) :
                unpack_frames(&decoder->vae, &tile, error, error_size);
            if (ok) {
                int index = tile_y * decoder->x_axis.count + tile_x;
                tiles[index] = tile.rgb;
            }
        }
    if (ok) ok = stitch_tiles(tiles, &decoder->y_axis, &decoder->x_axis,
                              frame_count, output, error, error_size);
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    if (!ok) h3_video_frames_free(output);
    return ok;
}
#endif

h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || latent_height < 1 ||
        latent_width < 1 || latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX) {
        fail(error, error_size, "invalid resident video VAE arguments");
        return NULL;
    }
    h3_video_vae_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        fail(error, error_size, "out of memory creating resident video VAE");
        return NULL;
    }
    decoder->latent_h = latent_height;
    decoder->latent_w = latent_width;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    int ok = load_latent_normalization(
        weight_directory, decoder->latent_mean, decoder->latent_std,
        error, error_size) &&
        tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                        &decoder->y_axis, error, error_size) &&
        tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                        &decoder->x_axis, error, error_size);
    if (ok && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: resident video VAE tiles %dx%d at %d pixels\n",
                decoder->x_axis.count, decoder->y_axis.count, tile_pixels);
    vae_context *vae = &decoder->vae;
    if (ok) {
        vae->latent_h = decoder->y_axis.length / SPATIAL_RATIO;
        vae->latent_w = decoder->x_axis.length / SPATIAL_RATIO;
        vae->latent_t = CHUNK_LATENT_TIME;
        vae->output_frames = FIRST_CHUNK_FRAMES;
        vae->patches = (uint32_t)(CHUNK_LATENT_TIME * vae->latent_h *
                                  vae->latent_w);
        vae->sequence = vae->patches + SUFFIX;
        vae->weights = h3_weight_store_open(weight_directory,
                                             error, error_size);
        if (vae->weights)
            vae->gpu = h3_gpu_create(shader_source_path, error, error_size);
        if (vae->gpu)
            h3_gpu_profile_set_label(vae->gpu, "resident video VAE decoder");
        ok = vae->weights && vae->gpu &&
             load_resident_weights(vae, progress, progress_opaque,
                                   error, error_size) &&
             prepare_rope(vae, error, error_size) &&
             allocate_activations(vae, error, error_size);
    }
    if (!ok) {
        h3_video_vae_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

static void free_tile_activations(vae_context *vae) {
    free_tensor(&vae->latent); free_tensor(&vae->post);
    free_tensor(&vae->patch_hidden); free_tensor(&vae->hidden);
    free_tensor(&vae->norm); free_tensor(&vae->qkv); free_tensor(&vae->query);
    free_tensor(&vae->key); free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->rope_cos);
    free_tensor(&vae->rope_sin); free_tensor(&vae->projected);
    free_tensor(&vae->norm_parts); free_tensor(&vae->act_parts);
    free_tensor(&vae->projected_next);
}

int h3_video_vae_decoder_reshape(h3_video_vae_decoder *decoder,
                                 int latent_height, int latent_width,
                                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!decoder || latent_height < 1 || latent_width < 1 ||
        latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX) {
        fail(error, error_size, "invalid resident video VAE reshape arguments");
        return 0;
    }
    if (decoder->latent_h == latent_height &&
        decoder->latent_w == latent_width) return 1;
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    if (!tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                         &y_axis, error, error_size) ||
        !tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                         &x_axis, error, error_size)) {
        tile_axis_free(&y_axis);
        tile_axis_free(&x_axis);
        return 0;
    }
    vae_context *vae = &decoder->vae;
    int tile_h = y_axis.length / SPATIAL_RATIO;
    int tile_w = x_axis.length / SPATIAL_RATIO;
    /* Tile activations and RoPE depend only on the tile geometry. The weights
     * are reused unchanged for every canvas. */
    if (tile_h != vae->latent_h || tile_w != vae->latent_w) {
        free_tile_activations(vae);
        vae->latent_h = tile_h;
        vae->latent_w = tile_w;
        vae->patches = (uint32_t)(CHUNK_LATENT_TIME * tile_h * tile_w);
        vae->sequence = vae->patches + SUFFIX;
        if (!prepare_rope(vae, error, error_size) ||
            !allocate_activations(vae, error, error_size)) {
            tile_axis_free(&y_axis);
            tile_axis_free(&x_axis);
            return 0;
        }
    }
    tile_axis_free(&decoder->y_axis);
    tile_axis_free(&decoder->x_axis);
    decoder->y_axis = y_axis;
    decoder->x_axis = x_axis;
    decoder->latent_h = latent_height;
    decoder->latent_w = latent_width;
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: resident video VAE tiles %dx%d at %d pixels\n",
                decoder->x_axis.count, decoder->y_axis.count, tile_pixels);
    return 1;
}

int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output || !output_frame_index ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid video VAE preview arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int chunk = chunks / 2;
    int local_frame = FIRST_CHUNK_FRAMES / 2;
    int output_frames = chunks * 17 + 5;
    int global_frame = chunk * 17 + local_frame;
    if (global_frame >= output_frames) global_frame = output_frames - 1;
    int ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, local_frame, output,
                                  error, error_size);
    if (ok) *output_frame_index = global_frame;
#ifdef H3_CUDA
    if (ok) finalize_video_pixels(output);
#endif
    return ok;
}

static int decoder_decode_all(h3_video_vae_decoder *decoder,
                              const float *normalized_latent, int latent_time,
                              h3_video_frames *output, uint8_t **rgb8,
                              h3_video_frames_ready ready, void *ready_opaque,
                              char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (rgb8) *rgb8 = NULL;
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid resident video VAE decode arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = decoder->latent_h * SPATIAL_RATIO;
    int pixel_w = decoder->latent_w * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = malloc(output_elements * sizeof(*final_rgb));
    float *temporal_overlap = malloc(5 * frame_elements *
                                     sizeof(*temporal_overlap));
    if (!final_rgb || !temporal_overlap) {
        free(final_rgb);
        free(temporal_overlap);
        fail(error, error_size, "out of memory retaining decoded video frames");
        return 0;
    }
    int ok = 1;
#ifdef H3_CUDA
    /* All chunks in one tile pipeline; the host thread blends and finalizes
     * each chunk while the GPU decodes the next tiles. */
    uint8_t *bytes = NULL;
    if (rgb8 && !(bytes = malloc(output_elements))) {
        free(final_rgb);
        free(temporal_overlap);
        fail(error, error_size, "out of memory converting decoded video frames");
        return 0;
    }
    video_sink_state state = {final_rgb, temporal_overlap, frame_elements, chunks,
                              bytes, ready, ready_opaque};
    vae_pipeline pipeline;
    memset(&pipeline, 0, sizeof(pipeline));
    pipeline.decoder = decoder;
    pipeline.latent = normalized_latent;
    pipeline.latent_time = latent_time;
    pipeline.first_chunk = 0;
    pipeline.chunk_count = chunks;
    pipeline.selected_frame = -1;
    pipeline.frame_count = FIRST_CHUNK_FRAMES;
    pipeline.sink = video_chunk_sink;
    pipeline.sink_state = &state;
    ok = pipeline_run(&pipeline, error, error_size);
    if (ok) {
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (!ok) {
        h3_video_frames_free(output);
        free(bytes);
    } else if (rgb8) {
        *rgb8 = bytes;
    }
    return ok;
#else
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, -1, &decoded, error, error_size);
        if (!ok) break;
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (!ok) h3_video_frames_free(output);
    if (ok && rgb8) {
        size_t count = (size_t)output->frames * output->height * output->width * 3;
        *rgb8 = malloc(count);
        if (!*rgb8) {
            h3_video_frames_free(output);
            fail(error, error_size, "out of memory converting decoded video frames");
            return 0;
        }
        for (size_t index = 0; index < count; index++) {
            float scaled = output->rgb[index] * 255.0f;
            if (scaled < 0.0f) scaled = 0.0f;
            if (scaled > 255.0f) scaled = 255.0f;
            (*rgb8)[index] = (uint8_t)lrintf(scaled);
        }
        if (ready) ready(ready_opaque, *rgb8, 0, output->frames);
    }
    return ok;
#endif
}

int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    return decoder_decode_all(decoder, normalized_latent, latent_time, output,
                              NULL, NULL, NULL, error, error_size);
}

int h3_video_vae_decoder_decode_rgb8(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, uint8_t **rgb8,
                        h3_video_frames_ready ready, void *ready_opaque,
                        char *error, size_t error_size) {
    if (rgb8) *rgb8 = NULL;
    if (!rgb8) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "invalid resident video VAE RGB24 output");
        return 0;
    }
    return decoder_decode_all(decoder, normalized_latent, latent_time, output,
                              rgb8, ready, ready_opaque, error, error_size);
}

void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder) {
    if (!decoder) return;
#ifdef H3_CUDA
    for (int slot = 0; slot < PIPELINE_SLOTS; slot++) {
        h3_gpu_event_free(decoder->readback[slot]);
        h3_gpu_host_free(decoder->pinned_latent[slot]);
        h3_gpu_host_free(decoder->pinned_rows[slot]);
    }
#endif
    cleanup(&decoder->vae);
    tile_axis_free(&decoder->y_axis);
    tile_axis_free(&decoder->x_axis);
    free(decoder);
}

static int decode_chunked(const char *weight_directory,
                          const char *shader_source_path,
                          const float *normalized_latent, int latent_time,
                          int latent_height, int latent_width,
                          const float *latent_mean, const float *latent_std,
                          int tile_pixels,
                          h3_video_vae_progress progress, void *progress_opaque,
                          h3_video_frames *output, char *error,
                          size_t error_size) {
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    int ok = tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                             &y_axis,
                             error, error_size) &&
             tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                             &x_axis,
                             error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: video VAE tiles %dx%d at %d pixels\n",
                x_axis.count, y_axis.count, tile_pixels);
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = y_axis.length / SPATIAL_RATIO;
    vae.latent_w = x_axis.length / SPATIAL_RATIO;
    vae.latent_t = CHUNK_LATENT_TIME;
    vae.output_frames = FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * vae.latent_h * vae.latent_w);
    vae.sequence = vae.patches + SUFFIX;
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (vae.weights)
        vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    ok = vae.weights && vae.gpu &&
         load_resident_weights(&vae, progress, progress_opaque,
                               error, error_size) &&
         prepare_rope(&vae, error, error_size) &&
         allocate_activations(&vae, error, error_size);
    int tile_count = y_axis.count * x_axis.count;
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = ok ? malloc(output_elements * sizeof(*final_rgb)) : NULL;
    float *temporal_overlap = ok ? malloc(5 * frame_elements *
                                          sizeof(*temporal_overlap)) : NULL;
    if (ok && (!final_rgb || !temporal_overlap)) {
        fail(error, error_size, "out of memory retaining decoded video frames");
        ok = 0;
    }
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
        if (!tiles) {
            fail(error, error_size, "out of memory retaining video VAE tiles");
            ok = 0;
            break;
        }
        for (int tile_y = 0; tile_y < y_axis.count && ok; tile_y++)
            for (int tile_x = 0; tile_x < x_axis.count && ok; tile_x++) {
                float *input = extract_latent_tile(normalized_latent,
                    latent_time, latent_height, latent_width, chunk * 5,
                    y_axis.starts[tile_y] / SPATIAL_RATIO,
                    x_axis.starts[tile_x] / SPATIAL_RATIO,
                    CHUNK_LATENT_TIME, vae.latent_h, vae.latent_w,
                    error, error_size);
                if (!input) {
                    ok = 0;
                    break;
                }
                free_tensor(&vae.latent);
                ok = prepare_input(&vae, input, latent_mean, latent_std,
                                   error, error_size) &&
                     run_resident_tile(&vae, vae.projected, 1, error, error_size);
                free(input);
                if (!ok) break;
                h3_video_frames tile;
                memset(&tile, 0, sizeof(tile));
                ok = unpack_frames(&vae, &tile, error, error_size);
                if (ok) {
                    int index = tile_y * x_axis.count + tile_x;
                    tiles[index] = tile.rgb;
                }
            }
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        if (ok) ok = stitch_tiles(tiles, &y_axis, &x_axis,
                                  FIRST_CHUNK_FRAMES, &decoded,
                                  error, error_size);
        for (int index = 0; index < tile_count; index++) free(tiles[index]);
        free(tiles);
        if (!ok) {
            h3_video_frames_free(&decoded);
            break;
        }
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    cleanup(&vae);
    tile_axis_free(&y_axis); tile_axis_free(&x_axis);
    return ok;
}

void h3_video_frames_free(h3_video_frames *frames) {
    if (!frames) return;
    free(frames->rgb);
    memset(frames, 0, sizeof(*frames));
}

int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_time < 2 ||
        (latent_time != 2 && (latent_time < CHUNK_LATENT_TIME ||
                              (latent_time - 2) % 5)) ||
        latent_height < 1 || latent_width < 1 ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX ||
        (uint32_t)(latent_height * latent_width) >
            (UINT32_MAX - SUFFIX) / CHUNK_LATENT_TIME) {
        fail(error, error_size, "invalid first-chunk video VAE arguments");
        return 0;
    }
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = latent_height;
    vae.latent_w = latent_width;
    vae.latent_t = latent_time;
    vae.output_frames = latent_time == 2 ? 5 : FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * latent_height * latent_width);
    vae.sequence = vae.patches + SUFFIX;
    float latent_mean[LATENT_CHANNELS], latent_std[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, latent_mean, latent_std,
                                   error, error_size)) return 0;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    if (latent_time == 2 &&
        (latent_height > TILE_PIXELS / SPATIAL_RATIO ||
         latent_width > TILE_PIXELS / SPATIAL_RATIO)) {
        fail(error, error_size,
             "the two-token diagnostic VAE path supports one spatial tile");
        return 0;
    }
    if (latent_time > CHUNK_LATENT_TIME ||
        latent_height > tile_pixels / SPATIAL_RATIO ||
        latent_width > tile_pixels / SPATIAL_RATIO) {
        int ok = decode_chunked(weight_directory, shader_source_path,
                                normalized_latent, latent_time, latent_height,
                                latent_width, latent_mean, latent_std,
                                tile_pixels, progress,
                                progress_opaque, output, error, error_size);
        if (!ok) h3_video_frames_free(output);
#ifdef H3_CUDA
        if (ok) finalize_video_pixels(output);
#endif
        return ok;
    }
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights) return 0;
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    int ok = vae.gpu &&
        load_input_weights(&vae, error, error_size) &&
        prepare_input(&vae, normalized_latent, latent_mean, latent_std,
                      error, error_size) &&
        prepare_rope(&vae, error, error_size) &&
        allocate_activations(&vae, error, error_size) &&
        run_decoder(&vae, progress, progress_opaque, error, error_size) &&
        unpack_frames(&vae, output, error, error_size);
    if (!ok) h3_video_frames_free(output);
#ifdef H3_CUDA
    if (ok) finalize_video_pixels(output);
#endif
    cleanup(&vae);
    return ok;
}
