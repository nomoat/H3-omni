/* Narrow test interface. Production inference still uses the ordinary C API. */
#include "h3_audio_vae.h"
#include "h3_dit.h"
#include "h3_video_vae.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    h3_dit *dit;
    h3_layout layout;
} h3_phase1_model;

static void progress(const char *phase, int done, int total, void *unused) {
    (void)unused;
    if (done == 0 || done == total || done % 10 == 0)
        fprintf(stderr, "CUDA %s %d/%d\n", phase, done, total);
}

h3_phase1_model *h3_phase1_create(const char *weights, const uint16_t *text, size_t tokens,
                                  int time, int height, int width, int audio_time, int frames,
                                  int steps, const float *video_sigmas, const float *audio_sigmas,
                                  char *error, size_t error_size) {
    if (steps < 1 || steps > H3_MAX_STEPS || !text || !tokens)
        return NULL;
    h3_phase1_model *model = calloc(1, sizeof(*model));
    if (!model)
        return NULL;
    h3_layout_spec spec = {(int)tokens, time, height, width, audio_time, frames, NULL, 0, NULL, 0};
    if (!h3_layout_build(&spec, &model->layout, error, error_size)) {
        free(model);
        return NULL;
    }
    h3_sigma_schedule sigmas = {.steps = steps};
    memcpy(sigmas.video, video_sigmas, (steps + 1) * sizeof(float));
    memcpy(sigmas.audio, audio_sigmas, (steps + 1) * sizeof(float));
    h3_text_embedding embedding = {0};
    embedding.tokens = tokens;
    embedding.width = H3_TEXT_HIDDEN_SIZE;
    embedding.values = (uint16_t *)text;
    model->dit = h3_dit_load_t2va(weights, "h3_shaders.metal", &embedding, &model->layout, &sigmas,
                                  50, 1, 0, 0, 1.0f, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, progress,
                                  NULL, error, error_size);
    if (!model->dit) {
        h3_layout_free(&model->layout);
        free(model);
        return NULL;
    }
    return model;
}

size_t h3_phase1_positions(h3_phase1_model *model, double *output, size_t count) {
    size_t need = model->layout.seq_len * 3;
    if (output && count >= need)
        memcpy(output, model->layout.positions, need * sizeof(double));
    return need;
}

size_t h3_phase1_tags(h3_phase1_model *model, int64_t *output, size_t count) {
    size_t need = model->layout.seq_len;
    if (output && count >= need) {
        for (size_t s = 0; s < model->layout.segment_count; ++s) {
            const h3_segment *segment = &model->layout.segments[s];
            int64_t tag = segment->kind == H3_SEG_TEXT ? 1 :
                          segment->kind == H3_SEG_AUDIO ? 2 : 0;
            for (size_t i = segment->start; i < segment->stop; ++i)
                output[i] = tag;
        }
    }
    return need;
}

int h3_phase1_forward(h3_phase1_model *model, int step, const float *video, const float *audio,
                      float *video_velocity, float *audio_velocity, char *error,
                      size_t error_size) {
    return h3_dit_forward(model->dit, step, video, audio, video_velocity, audio_velocity, error,
                          error_size);
}

void h3_phase1_free(h3_phase1_model *model) {
    if (!model)
        return;
    h3_dit_free(model->dit);
    h3_layout_free(&model->layout);
    free(model);
}

int h3_phase1_reset(h3_phase1_model *model, char *error, size_t error_size) {
    return h3_dit_reset_run(model->dit, NULL, 0, NULL, 0, error, error_size);
}

static void text_progress(int done, int total, void *unused) {
    (void)unused;
    if (done == 0 || done == total || done % 10 == 0)
        fprintf(stderr, "CUDA Qwen %d/%d\n", done, total);
}

int h3_phase1_text(const char *weights, const uint32_t *ids, size_t tokens, uint16_t *output,
                   char *error, size_t error_size) {
    h3_text_embedding embedding = {0};
    int ok = h3_text_encode_bf16(weights, "h3_shaders.metal", ids, tokens, text_progress, NULL,
                                 &embedding, error, error_size);
    if (ok)
        memcpy(output, embedding.values, tokens * 5120 * sizeof(uint16_t));
    h3_text_embedding_free(&embedding);
    return ok;
}

int h3_phase1_video(const char *weights, const float *latent, int time, int height, int width,
                    float *output, size_t capacity, int *shape, char *error, size_t error_size) {
    h3_video_frames frames = {0};
    int ok = h3_video_vae_decode(weights, "h3_shaders.metal", latent, time, height, width, NULL,
                                 NULL, &frames, error, error_size);
    size_t count = (size_t)frames.frames * frames.height * frames.width * 3;
    if (ok && count > capacity) {
        snprintf(error, error_size, "video output exceeds test buffer");
        ok = 0;
    }
    if (ok) {
        shape[0] = frames.frames;
        shape[1] = frames.height;
        shape[2] = frames.width;
        memcpy(output, frames.rgb, count * sizeof(float));
    }
    h3_video_frames_free(&frames);
    return ok;
}

int h3_phase1_audio(const char *weights, const float *latent, int time, float *output,
                    size_t capacity, int *shape, char *error, size_t error_size) {
    h3_audio_waveform waveform = {0};
    int ok = h3_audio_vae_decode(weights, "h3_shaders.metal", latent, time, NULL, NULL, &waveform,
                                 error, error_size);
    size_t count = (size_t)waveform.channels * waveform.samples;
    if (ok && count > capacity) {
        snprintf(error, error_size, "audio output exceeds test buffer");
        ok = 0;
    }
    if (ok) {
        shape[0] = waveform.channels;
        shape[1] = waveform.samples;
        shape[2] = waveform.sample_rate;
        memcpy(output, waveform.pcm, count * sizeof(float));
    }
    h3_audio_waveform_free(&waveform);
    return ok;
}
