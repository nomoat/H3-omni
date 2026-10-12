#include "h3_live.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static float shifted(int index, int states, float shift) {
    float step = -1.0f / (float)(states - 1);
    /* torch.linspace computes the latter half from the terminal endpoint. */
    float base = index < states / 2 ? fmaf((float)index, step, 1.0f) :
                 -(states - index - 1) * step;
    return shift * base / (1.0f + (shift - 1.0f) * base);
}

int h3_live_schedule_build(h3_sigma_schedule *s) {
    if (!s) return 0;
    memset(s, 0, sizeof(*s));
    const int indices[] = {0, 16, 33, 49};
    s->steps = 4; /* Three noisy evaluations, then a clean cache evaluation. */
    for (int i = 0; i < 4; i++) {
        s->video[i] = shifted(indices[i], 50, 12);
        s->audio[i] = shifted(indices[i], 50, 3);
    }
    return 1;
}

static void teacher_schedule(h3_sigma_schedule *s) {
    memset(s, 0, sizeof(*s));
    s->steps = 9;
    for (int i = 0; i < 10; i++) {
        s->video[i] = shifted(i, 10, 12);
        s->audio[i] = shifted(i, 10, 3);
    }
}

static int audio_boundary(int frame) {
    /* Exact round-half-even on the global 40 Hz / 24 fps timeline. */
    int n = frame * 5, q = n / 3, r = n % 3;
    return q + (r == 2);
}

static double video_time(int index) {
    int cycles = index / 5, remainder = index % 5;
    int ticks = cycles*17 + (remainder ? remainder*4-3 : 0);
    return (double)(ticks*5) / 3.0;
}

/* Native channel-major slices. Audio has 64 channels (32 x stereo). */
static void slice(float *out, const float *in, int channels, int total,
                   int start, int count, size_t area) {
    for (int c = 0; c < channels; c++)
        memcpy(out + (size_t)c * count * area,
               in + ((size_t)c * total + start) * area,
               (size_t)count * area * sizeof(float));
}

static void insert(float *out, const float *in, int channels, int total,
                    int start, int count, size_t area) {
    for (int c = 0; c < channels; c++)
        memcpy(out + ((size_t)c * total + start) * area,
               in + (size_t)c * count * area,
               (size_t)count * area * sizeof(float));
}

static int layout_build(h3_layout *layout, const h3_text_embedding *text,
                         int vt, int h, int w, int at, int video_start,
                         int request_video_start, int audio_start,
                         int teacher, int previous_audio_t,
                         char *error, size_t capacity) {
    h3_layout_ref ref = {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 40};
    h3_layout_spec spec = {(int)text->tokens, vt ? vt : 1, h, w, at, 5,
                           NULL, 0, previous_audio_t ? &ref : NULL,
                           previous_audio_t ? 1 : 0};
    if (!h3_layout_build(&spec, layout, error, capacity)) return 0;
    int origin = (int)text->tokens;
    if (!vt) {
        h3_segment *last = &layout->segments[layout->segment_count - 1];
        last->stop = last->start;
        layout->seq_len = last->start;
        layout->img_target_rows = 0;
        layout->signature[1] = 0;
    }
    for (size_t i = 0; i < layout->segment_count; i++) {
        h3_segment *seg = &layout->segments[i];
        for (size_t row = seg->start; row < seg->stop; row++) {
            size_t j = row - seg->start;
            double t;
            switch (seg->kind) {
            case H3_SEG_TEXT:
                t = j + (teacher ? 0 : video_time(request_video_start)); break;
            case H3_SEG_REF_AUDIO:
                t = origin + previous_audio_t - 40 + (j % 40); break;
            case H3_SEG_AUDIO:
                t = origin + (teacher ? previous_audio_t : audio_start) + (j % at); break;
            case H3_SEG_VIDEO:
                t = origin + video_time(video_start + (int)(j / ((h/2)*(w/2)))); break;
            default: t = layout->positions[row].t; break;
            }
            layout->positions[row].t = t;
        }
    }
    return 1;
}

static int rebind(h3_dit *dit, const h3_text_embedding *text,
                    h3_layout *layout, h3_sigma_schedule *schedule,
                    const float *reference, char *error, size_t capacity) {
    return h3_dit_rebind(dit, text, layout, schedule, 50, 1, 0, 1.0f,
                           NULL, 0, reference, layout->audio_cond_rows * 32,
                           NULL, NULL, error, capacity) == 1;
}

static void update(float *state, const float *velocity, size_t n,
                     float sigma, float next) {
    float ratio = next / sigma, one_minus = 1.0f - ratio;
    for (size_t i = 0; i < n; i++) {
        float sigma_t = 1.0f - (1.0f - sigma);
        float clean = state[i] + sigma_t * velocity[i];
        state[i] = ratio * state[i] + one_minus * clean;
    }
}

static int renorm(float *video, int t, int h, int w,
                   float *anchor_mean, float *anchor_std, int first) {
    size_t elements = (size_t)24 * t * h * w, rows = elements / 96;
    float *packed = malloc(elements * sizeof(float));
    if (!packed || !h3_dit_patchify_video(video, 24, t, h, w, packed, elements)) {
        free(packed); return 0;
    }
    for (int feature = 0; feature < 96; feature++) {
        double sum = 0, variance = 0;
        for (size_t row = 0; row < rows; row++) sum += packed[row*96 + feature];
        float mean = (float)(sum / rows);
        for (size_t row = 0; row < rows; row++) {
            double delta = packed[row*96 + feature] - (double)mean;
            variance += delta * delta;
        }
        float std = fmaxf((float)sqrt(variance / rows), 1e-6f);
        if (first) { anchor_mean[feature] = mean; anchor_std[feature] = std; }
        else for (size_t row = 0; row < rows; row++)
            packed[row*96 + feature] = (packed[row*96 + feature] - mean) /
                std * anchor_std[feature] + anchor_mean[feature];
    }
    int ok = h3_dit_unpatchify_video(packed, 24, t, h, w, video, elements);
    free(packed);
    return ok;
}

typedef struct {
    int vt, at, video_start, audio_start, frame_start;
    float *video, *noise, *milestones[3];
} live_request;

int h3_live_generate(h3_dit *dit, const h3_text_embedding *text,
                     int video_t, int height, int width, int audio_t,
                     uint64_t seed, float *video, float *audio,
                     h3_dit_progress progress, void *opaque,
                     char *error, size_t capacity) {
    int requests = (video_t - 2) / 35;
    if (!dit || !text || !video || !audio || video_t != 2 + requests * 35 || requests < 1) {
        snprintf(error, capacity, "live mode requires 124, 243, 362, ... frames (5-second requests)");
        return 0;
    }
    live_request *req = calloc((size_t)requests, sizeof(*req));
    float *v = NULL, *a = NULL, *vv = NULL, *av = NULL;
    float *milestones[3] = {0};
    float *official_video = NULL, *official_audio = NULL, *audio_rows = NULL;
    float reference[64*40], reference_rows[64*40];
    float mean[96], std[96];
    h3_layout layout = {0};
    h3_sigma_schedule schedule;
    size_t area = (size_t)height * width;
    int ok = 0, chunk = 0;
    double begin = now();
#define REQUIRE(test) do { if (!(test)) goto done; } while (0)
    REQUIRE(req);
    official_video = malloc(24 * 37 * area * sizeof(float));
    official_audio = malloc(64 * 207 * sizeof(float));
    audio_rows = malloc(64 * 207 * sizeof(float));
    v = malloc(24 * 12 * area * sizeof(float));
    vv = malloc(24 * 12 * area * sizeof(float));
    a = malloc(64 * 207 * sizeof(float));
    av = malloc(64 * 207 * sizeof(float));
    for (int j=0;j<3;j++) milestones[j] = malloc(64*207*sizeof(float));
    REQUIRE(official_video && official_audio && audio_rows && v && vv && a && av &&
            milestones[0] && milestones[1] && milestones[2]);
    REQUIRE(h3_dit_live_control(dit, 0, 0, 0, 1, 0, 0, error, capacity));
    teacher_schedule(&schedule);
    for (int r = 0; r < requests; r++) {
        live_request *p = &req[r];
        p->video_start = r ? 2 + r * 35 : 0;
        p->frame_start = r ? 5 + r * 119 : 0;
        p->audio_start = audio_boundary(p->frame_start);
        p->vt = r ? 35 : 37;
        p->at = audio_boundary(5 + (r+1)*119) - p->audio_start;
        p->video = malloc((size_t)24 * p->vt * area * sizeof(float));
        p->noise = malloc((size_t)64 * p->at * sizeof(float));
        for (int j = 0; j < 3; j++) p->milestones[j] = malloc((size_t)64*p->at*sizeof(float));
        REQUIRE(p->video && p->noise && p->milestones[0] && p->milestones[1] && p->milestones[2]);
        /* One native generator shared by video and audio, as in TaoMate.
         * The native RNG is deterministic but differs from torch.randn. */
        h3_rng rng;
        h3_rng_seed(&rng, seed + (uint64_t)r);
        h3_rng_fill_normal(&rng, official_video, 24 * 37 * area);
        h3_rng_fill_normal(&rng, audio_rows, 64 * 207);
        REQUIRE(h3_dit_unpack_audio(audio_rows, 32, 207, official_audio, 64*207));
        slice(p->video, official_video, 24, 37, r ? 2 : 0, p->vt, area);
        slice(p->noise, official_audio, 64, 207, 207 - p->at, p->at, 1);
        memcpy(a, p->noise, (size_t)64*p->at*sizeof(float));
        if (r) {
            slice(reference, req[r-1].milestones[2], 64, req[r-1].at,
                    req[r-1].at-40, 40, 1);
            REQUIRE(h3_dit_pack_audio(reference, 32, 40, reference_rows, 64*40));
        }
        REQUIRE(layout_build(&layout, text, 0, height, width, p->at, 0, 0, 0,
                              1, r ? req[r-1].at : 0, error, capacity));
        REQUIRE(rebind(dit, text, &layout, &schedule, r ? reference_rows : NULL, error, capacity));
        h3_layout_free(&layout);
        for (int step = 0; step < 9; step++) {
            REQUIRE(h3_dit_forward(dit, step, v, a, vv, av, error, capacity));
            update(a, av, (size_t)64*p->at, schedule.audio[step], schedule.audio[step+1]);
            if ((step+1)%3 == 0)
                memcpy(p->milestones[(step+1)/3-1], a, (size_t)64*p->at*sizeof(float));
            if (progress) progress("live audio teacher", r*9+step+1, requests*9, opaque);
        }
        insert(audio, p->milestones[2], 64, audio_t, p->audio_start, p->at, 1);
    }
    double teacher_done = now();
    fprintf(stderr, "h3: live audio teacher: %d requests, %d forwards, %.3fs\n",
            requests, requests*9, teacher_done-begin);
    REQUIRE(h3_dit_live_control(dit, 1, 1, 0, 0, 0, 0, error, capacity));
    h3_live_schedule_build(&schedule);
    for (int r = 0; r < requests; r++) {
        live_request *p = &req[r];
        if (r && r%12 == 0)
            REQUIRE(h3_dit_live_control(dit, 1, 1, 0, 0, 0, 1, error, capacity));
        int vs = 0, as = 0, fs = 0;
        for (int phase = 0; phase < 4; phase++, chunk++) {
            int groups = phase == 3 ? 1 : 2;
            int vt = groups*5 + (r == 0 && phase == 0 ? 2 : 0);
            int frames = groups*17 + (r == 0 && phase == 0 ? 5 : 0);
            int ae = audio_boundary(p->frame_start + fs + frames) - p->audio_start;
            int at = ae - as;
            size_t vn = (size_t)24*vt*area, an = (size_t)64*at;
            slice(v, p->video, 24, p->vt, vs, vt, area);
            slice(a, p->noise, 64, p->at, as, at, 1);
            REQUIRE(layout_build(&layout, text, vt, height, width, at,
                       p->video_start+vs, p->video_start, p->audio_start+as, 0, 0, error, capacity));
            REQUIRE(rebind(dit, text, &layout, &schedule, NULL, error, capacity));
            h3_layout_free(&layout);
            double chunk_begin = now();
            for (int j=0;j<3;j++) slice(milestones[j],p->milestones[j],64,p->at,as,at,1);
            if (!getenv("H3_LIVE_HOST_SAMPLER")) {
                REQUIRE(h3_dit_live_chunk(dit,v,a,(const float *const *)milestones,error,capacity));
            } else for (int step = 0; step < 3; step++) {
                REQUIRE(h3_dit_forward(dit, step, v, a, vv, av, error, capacity));
                update(v, vv, vn, schedule.video[step], schedule.video[step+1]);
                /* Audio state at each boundary is exactly the teacher milestone. */
                slice(a, p->milestones[step], 64, p->at, as, at, 1);
                if (progress) progress("live denoise", chunk*4+step+1, requests*16, opaque);
            }
            REQUIRE(renorm(v, vt, height, width, mean, std, chunk == 0));
            REQUIRE(h3_dit_live_control(dit, 1, 1, 1, 0, 0, 0, error, capacity));
            REQUIRE(h3_dit_forward(dit, 3, v, a, vv, av, error, capacity));
            REQUIRE(h3_dit_live_control(dit, 1, 1, 0, 0, 1, 0, error, capacity));
            insert(video, v, 24, video_t, p->video_start+vs, vt, area);
            for (size_t i = 0; i < vn; i++) REQUIRE(isfinite(v[i]));
            for (size_t i = 0; i < an; i++) REQUIRE(isfinite(a[i]));
            if (progress) progress("live denoise", (chunk+1)*4, requests*16, opaque);
            fprintf(stderr, "h3: live request %d chunk %d: %d frames, 3 noisy + 1 clean, %.3fs\n",
                    r+1, phase+1, frames, now()-chunk_begin);
            vs += vt; as = ae; fs += frames;
        }
    }
    fprintf(stderr, "h3: live DiT total %.3fs (audio %.3fs, video %.3fs), %d clean commits\n",
            now()-begin, teacher_done-begin, now()-teacher_done, chunk);
    ok = 1;
done:
    h3_layout_free(&layout);
    if (req) for (int r = 0; r < requests; r++) {
        free(req[r].video); free(req[r].noise);
        for (int j = 0; j < 3; j++) free(req[r].milestones[j]);
    }
    for (int j=0;j<3;j++) free(milestones[j]);
    free(req); free(v); free(vv); free(a); free(av);
    free(official_video); free(official_audio); free(audio_rows);
    if (!ok && error && capacity && !*error)
        snprintf(error, capacity, "live generation failed: allocation or non-finite latent");
    return ok;
#undef REQUIRE
}
