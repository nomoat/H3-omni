#ifndef H3_LIVE_H
#define H3_LIVE_H
#include "h3_dit.h"

/* Released TaoMate-H3 rank-128 adapter and Stage3 sampling contract. */
int h3_live_config_read(const char *directory, int *enabled,
                         char *error, size_t error_size);
int h3_live_schedule_build(h3_sigma_schedule *schedule);
int h3_live_generate(h3_dit *dit, const h3_text_embedding *text,
                     int video_t, int height, int width, int audio_t,
                     uint64_t seed, float *video, float *audio,
                     h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size);

/* Incremental latent session. The caller owns the resident DiT and embedding,
 * which must outlive the session and cannot serve another request meanwhile.
 * Each next() finishes one request before returning; no future audio/video
 * is generated. KV, first-chunk statistics and the previous audio tail persist.
 * Decoding/transport are caller responsibilities. A failed session is terminal. */
typedef struct h3_live_session h3_live_session;
typedef struct {
    int request_index, frame_start, frame_count;
    int video_start, video_t, audio_start, audio_t;
} h3_live_part;
h3_live_session *h3_live_session_new(h3_dit *dit,
    const h3_text_embedding *text, int latent_height, int latent_width,
    uint64_t seed, char *error, size_t error_size);
int h3_live_session_next(h3_live_session *session,
    float *video, size_t video_capacity, float *audio, size_t audio_capacity,
    h3_live_part *part, h3_dit_progress progress, void *opaque,
    char *error, size_t error_size);
void h3_live_session_free(h3_live_session *session);

int h3_dit_live_chunk(h3_dit *dit, float *video, float *audio,
                       const float *const milestones[3], char *error, size_t error_size);

/* Internal session controls. History survives rebind, but never a new run.
 * A clean forward stages all layers; commit publishes them only on success. */
int h3_dit_live_control(h3_dit *dit, int adapter, int attention, int stage,
                        int reset, int commit, int clear_audio,
                        char *error, size_t error_size);
#endif
