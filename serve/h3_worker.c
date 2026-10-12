/* Resident generation worker for the HTTP video server (serve/server.py).
 *
 * Loads the model once, keeps every weight resident, then reads one request
 * per stdin line and writes one JSON object per line to stdout:
 *
 *   request: id TAB width TAB height TAB frames TAB steps TAB seed TAB
 *            output_path TAB prompt
 *   reply:   {"event":"ready",...} once after loading, then per request
 *            {"event":"done","id":...,"seconds":...} or
 *            {"event":"error","id":...,"message":...}
 *
 * A failed request does not stop the worker. The prompt is the last field, so
 * it may contain any byte except TAB, CR and LF; the server replaces those.
 * H3_GPUS selects the GPU count, as for the CLI.
 *
 * Usage: h3_worker MODEL_DIR */
#include "h3.h"
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void print_json_string(const char *text) {
    putchar('"');
    for (const unsigned char *c = (const unsigned char *)text; c && *c; c++) {
        if (*c == '"' || *c == '\\') printf("\\%c", *c);
        else if (*c < 0x20) printf("\\u%04x", *c);
        else putchar(*c);
    }
    putchar('"');
}

typedef struct { const char *id; char cancel[4096]; } live_progress;
static int live_update(const char *phase, int completed, int total, void *opaque) {
    live_progress *p=opaque;
    if (!access(p->cancel,F_OK)) return 1;
    if (!strcmp(phase,"live part")) {
        printf("{\"event\":\"part\",\"id\":"); print_json_string(p->id);
        printf(",\"part\":%d,\"parts\":%d}\n",completed,total); fflush(stdout);
    }
    return 0;
}

static void reply_error(const char *id, const char *message) {
    printf("{\"event\":\"error\",\"id\":");
    print_json_string(id);
    printf(",\"message\":");
    print_json_string(message ? message : "unknown error");
    printf("}\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    double started = now();
    h3_ctx *ctx = h3_load_dir(argv[1]);
    if (!ctx) {
        printf("{\"event\":\"fatal\",\"message\":");
        print_json_string(h3_last_error(NULL));
        printf("}\n");
        return 1;
    }
    h3_cache_set_enabled(ctx, 1);
    h3_cache_set_resident_weights(ctx, 1);
    printf("{\"event\":\"ready\",\"seconds\":%.3f}\n", now() - started);
    fflush(stdout);

    static char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        size_t length = strcspn(line, "\r\n");
        int complete = line[length] != '\0';
        line[length] = '\0';
        if (!complete) {
            /* Discard the rest of an over-long line. */
            int c;
            while ((c = getchar()) != EOF && c != '\n') {}
            reply_error("", "request line too long");
            continue;
        }
        if (!line[0]) continue;
        char *fields[8];
        char *cursor = line;
        int count = 0;
        for (; count < 7; count++) {
            fields[count] = cursor;
            char *tab = strchr(cursor, '\t');
            if (!tab) break;
            *tab = '\0';
            cursor = tab + 1;
        }
        if (count != 7) {
            reply_error(fields[0], "malformed request line");
            continue;
        }
        fields[7] = cursor;
        h3_params params = H3_PARAMS_DEFAULT;
        params.width = atoi(fields[1]);
        params.height = atoi(fields[2]);
        params.frames = atoi(fields[3]);
        params.steps = atoi(fields[4]);
        params.seed = strtoull(fields[5], NULL, 10);
        params.output_path = fields[6];
        double request_started = now();
        /* Negative frame count denotes the bounded live request count.
         * This private protocol is written only by the authenticated server. */
        int live=params.frames<0;
        live_progress progress={.id=fields[0]};
        snprintf(progress.cancel,sizeof(progress.cancel),"%s.cancel",params.output_path);
        if (live) { params.on_progress=live_update; params.callback_opaque=&progress; }
        h3_result *result = live ? h3_generate_live(ctx,fields[7],&params,-params.frames) :
                                  h3_generate(ctx, fields[7], &params);
        if (!result) {
            reply_error(fields[0], h3_last_error(ctx));
            continue;
        }
        h3_result_free(result);
        printf("{\"event\":\"done\",\"id\":");
        print_json_string(fields[0]);
        printf(",\"seconds\":%.3f}\n", now() - request_started);
        fflush(stdout);
    }
    h3_free(ctx);
    return 0;
}
