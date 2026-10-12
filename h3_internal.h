#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "h3.h"
#include "h3_host.h"
#include "h3_fasth3.h"

#include <stdarg.h>

struct h3_dit;
struct h3_video_vae_decoder;
struct h3_tokenizer;
struct h3_text_encoder;
struct h3_audio_vae_decoder;

struct h3_ctx {
    h3_fasth3_config fasth3;
    int live;
    char *model_dir;
    char error[512];
    /* Why ordered references cannot run, when Ref2VA/ exists but is incomplete. */
    char ref2va_unavailable[512];
    h3_device_info device;
    h3_model_info model;
    int cache_enabled;
    char *conditioning_key;
    size_t conditioning_tokens;
    size_t conditioning_width;
    uint16_t *conditioning_values;
    uint8_t *conditioning_tags;
    float *conditioning_video_rows;
    size_t conditioning_video_elements;
    float *conditioning_audio_rows;
    size_t conditioning_audio_elements;
    h3_layout_ref *conditioning_references;
    size_t conditioning_reference_count;
    int conditioning_present;
    char *dit_key;
    struct h3_dit *dit;
    char *video_decoder_key;
    struct h3_video_vae_decoder *video_decoder;
    /* Serving residency (with cache_enabled): request-independent weights
     * stay loaded and only request state is rebuilt for a new request. */
    int resident_weights;
    char *dit_weights_key;
    char *tokenizer_key;
    struct h3_tokenizer *tokenizer;
    char *text_encoder_key;
    struct h3_text_encoder *text_encoder;
    char *audio_decoder_key;
    struct h3_audio_vae_decoder *audio_decoder;
};

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
