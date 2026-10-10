/* Serving check: run several different requests in one process with resident
 * weights. Each request writes its MP4 and, through H3_DUMP_DIR, the exact
 * float32 latents and decoded media for comparison with fresh one-shot runs.
 *
 * Usage: h3_serve_check MODEL_DIR OUTPUT_DIR REQUESTS_FILE
 * Each REQUESTS_FILE line: name|width|height|frames|steps|seed|prompt
 * One JSON object per request is printed to stdout. */
#include "h3.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define MAX_PHASES 32

typedef struct {
    double start;
    char names[MAX_PHASES][48];
    double first[MAX_PHASES];
    double last[MAX_PHASES];
    int count;
} phase_log;

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int on_progress(const char *phase, int completed, int total,
                       void *opaque) {
    (void)completed;
    (void)total;
    phase_log *log = opaque;
    double at = now() - log->start;
    for (int index = 0; index < log->count; index++) {
        if (!strcmp(log->names[index], phase)) {
            log->last[index] = at;
            return 0;
        }
    }
    if (log->count < MAX_PHASES) {
        snprintf(log->names[log->count], sizeof(log->names[0]), "%s", phase);
        log->first[log->count] = at;
        log->last[log->count] = at;
        log->count++;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL_DIR OUTPUT_DIR REQUESTS_FILE\n",
                argv[0]);
        return 2;
    }
    FILE *requests = fopen(argv[3], "r");
    if (!requests) {
        fprintf(stderr, "cannot open %s: %s\n", argv[3], strerror(errno));
        return 2;
    }
    double load_started = now();
    h3_ctx *ctx = h3_load_dir(argv[1]);
    if (!ctx) {
        fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
        return 1;
    }
    h3_cache_set_enabled(ctx, 1);
    h3_cache_set_resident_weights(ctx, 1);
    printf("{\"event\":\"context\",\"seconds\":%.6f}\n", now() - load_started);
    fflush(stdout);
    char line[8192];
    int status = 0;
    while (fgets(line, sizeof(line), requests)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;
        char *fields[7];
        char *cursor = line;
        int count = 0;
        for (; count < 6; count++) {
            fields[count] = cursor;
            char *bar = strchr(cursor, '|');
            if (!bar) break;
            *bar = '\0';
            cursor = bar + 1;
        }
        if (count != 6) {
            fprintf(stderr, "malformed request line\n");
            status = 2;
            break;
        }
        fields[6] = cursor;
        char directory[4096], output[4096];
        snprintf(directory, sizeof(directory), "%s/%s", argv[2], fields[0]);
        snprintf(output, sizeof(output), "%s/%s/output.mp4", argv[2],
                 fields[0]);
        if (mkdir(directory, 0755) && errno != EEXIST) {
            fprintf(stderr, "cannot create %s\n", directory);
            status = 2;
            break;
        }
        setenv("H3_DUMP_DIR", directory, 1);
        h3_params params = H3_PARAMS_DEFAULT;
        params.width = atoi(fields[1]);
        params.height = atoi(fields[2]);
        params.frames = atoi(fields[3]);
        params.steps = atoi(fields[4]);
        params.seed = strtoull(fields[5], NULL, 10);
        params.output_path = output;
        phase_log log;
        memset(&log, 0, sizeof(log));
        params.on_progress = on_progress;
        params.callback_opaque = &log;
        log.start = now();
        h3_result *result = h3_generate(ctx, fields[6], &params);
        double seconds = now() - log.start;
        if (!result) {
            fprintf(stderr, "h3 %s: %s\n", fields[0], h3_last_error(ctx));
            status = 1;
            break;
        }
        h3_result_free(result);
        h3_cache_info info;
        h3_cache_get_info(ctx, &info);
        printf("{\"event\":\"request\",\"name\":\"%s\",\"seconds\":%.6f,"
               "\"text_encoder\":%d,\"audio_decoder\":%d,"
               "\"video_decoder\":%d,\"dit\":%d,\"phases\":[",
               fields[0], seconds, info.text_encoder, info.audio_decoder,
               info.video_decoder, info.prepared_dit);
        for (int index = 0; index < log.count; index++)
            printf("%s{\"phase\":\"%s\",\"first\":%.6f,\"last\":%.6f}",
                   index ? "," : "", log.names[index], log.first[index],
                   log.last[index]);
        printf("]}\n");
        fflush(stdout);
    }
    fclose(requests);
    unsetenv("H3_DUMP_DIR");
    h3_free(ctx);
    return status;
}
