/* Read-only workload probe: link the accepted model objects without edits.
 * Callback times are coarse CPU wall times, not GPU kernel measurements. */
#define _POSIX_C_SOURCE 200809L
#include "h3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

typedef struct { int request; double start; } probe;

static int progress(const char *phase, int completed, int total, void *opaque) {
    probe *p = opaque;
    /* Phase names are fixed internal strings, containing no JSON escapes. */
    printf("{\"event\":\"progress\",\"request\":%d,\"phase\":\"%s\","
           "\"completed\":%d,\"total\":%d,\"seconds\":%.9f}\n",
           p->request, phase, completed, total, now() - p->start);
    fflush(stdout);
    return 0;
}

static char *read_prompt(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *s = calloc(16385, 1);
    if (!s) { fclose(f); return NULL; }
    size_t n = fread(s, 1, 16384, f);
    int extra = fgetc(f);
    fclose(f);
    if (!n || extra != EOF) { free(s); return NULL; }
    while (n && (s[n-1] == '\n' || s[n-1] == '\r')) s[--n] = 0;
    return s;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL_ROOT PROMPT_A PROMPT_B OUTPUT_DIR\n", argv[0]);
        return 2;
    }
    char *prompts[] = {read_prompt(argv[2]), read_prompt(argv[3])};
    if (!prompts[0] || !prompts[1]) return 2;
    double started = now();
    h3_ctx *ctx = h3_load_dir(argv[1]);
    if (!ctx) { fprintf(stderr, "%s\n", h3_last_error(NULL)); return 1; }
    printf("{\"event\":\"context_load\",\"seconds\":%.9f}\n", now() - started);
    h3_cache_set_enabled(ctx, 1);
    const char *names[] = {"first_request", "cached_same_prompt_new_seed", "changed_prompt_new_seed"};
    for (int i = 0; i < 3; ++i) {
        char output[4096];
        if (snprintf(output, sizeof(output), "%s/request-%d.mp4", argv[4], i) >= (int)sizeof(output)) return 2;
        h3_params params = H3_PARAMS_DEFAULT;
        params.seed = 7 + i;
        params.output_path = output;
        params.on_progress = progress;
        probe p = {i, now()};
        params.callback_opaque = &p;
        h3_result *result = h3_generate(ctx, prompts[i == 2], &params);
        double elapsed = now() - p.start;
        if (!result) { fprintf(stderr, "%s\n", h3_last_error(ctx)); h3_free(ctx); return 1; }
        h3_cache_info cache;
        h3_cache_get_info(ctx, &cache);
        printf("{\"event\":\"result\",\"request\":%d,\"mode\":\"%s\","
               "\"seconds\":%.9f,\"width\":%d,\"height\":%d,\"frames\":%d,"
               "\"fps\":%d,\"seed\":%llu,\"cached_dit\":%d,\"cached_video_decoder\":%d}\n",
               i, names[i], elapsed, result->width, result->height, result->frames,
               result->fps, (unsigned long long)result->seed, cache.prepared_dit, cache.video_decoder);
        fflush(stdout);
        h3_result_free(result);
    }
    h3_free(ctx);
    free(prompts[0]); free(prompts[1]);
    return 0;
}
