/* CUPTI kernel tracer loaded through CUDA_INJECTION64_PATH.
 *
 * Records every kernel, memcpy and memset with GPU start/end timestamps to
 * the text file named by H3_CUPTI_OUT. The first line maps the CUPTI clock to
 * CLOCK_MONOTONIC so records can be matched with h3_serve_check phases.
 *
 * Lines: C cupti_ns monotonic_ns
 *        K start end stream grid_x grid_y grid_z block_x block_y block_z name
 *        M start end bytes copy_kind
 *        S start end bytes
 *        A start end domain cbid thread name  (H3_CUPTI_API=1: host API calls) */
#include <cupti.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BUFFER_SIZE (8u << 20)

static FILE *trace;
static pthread_mutex_t trace_lock = PTHREAD_MUTEX_INITIALIZER;

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                      size_t *max_records) {
    *buffer = aligned_alloc(8, BUFFER_SIZE);
    *size = *buffer ? BUFFER_SIZE : 0;
    *max_records = 0;
}

static void CUPTIAPI buffer_completed(CUcontext context, uint32_t stream,
                                      uint8_t *buffer, size_t size,
                                      size_t valid) {
    (void)context;
    (void)stream;
    (void)size;
    CUpti_Activity *record = NULL;
    pthread_mutex_lock(&trace_lock);
    while (cuptiActivityGetNextRecord(buffer, valid, &record) ==
           CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const CUpti_ActivityKernel9 *k = (const void *)record;
            fprintf(trace, "K %llu %llu %u %d %d %d %d %d %d %s\n",
                    (unsigned long long)k->start, (unsigned long long)k->end,
                    k->streamId, k->gridX, k->gridY, k->gridZ, k->blockX,
                    k->blockY, k->blockZ, k->name ? k->name : "?");
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            const CUpti_ActivityMemcpy6 *m = (const void *)record;
            fprintf(trace, "M %llu %llu %llu %u\n",
                    (unsigned long long)m->start, (unsigned long long)m->end,
                    (unsigned long long)m->bytes, m->copyKind);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ||
                   record->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
            const CUpti_ActivityAPI *a = (const void *)record;
            int runtime = record->kind == CUPTI_ACTIVITY_KIND_RUNTIME;
            const char *name = NULL;
            cuptiGetCallbackName(runtime ? CUPTI_CB_DOMAIN_RUNTIME_API
                                         : CUPTI_CB_DOMAIN_DRIVER_API,
                                 a->cbid, &name);
            fprintf(trace, "A %llu %llu %c %u %u %s\n",
                    (unsigned long long)a->start, (unsigned long long)a->end,
                    runtime ? 'R' : 'D', a->cbid, a->threadId,
                    name ? name : "?");
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            const CUpti_ActivityMemset4 *m = (const void *)record;
            fprintf(trace, "S %llu %llu %llu\n",
                    (unsigned long long)m->start, (unsigned long long)m->end,
                    (unsigned long long)m->bytes);
        }
    }
    pthread_mutex_unlock(&trace_lock);
    free(buffer);
}

static void finish(void) {
    cuptiActivityFlushAll(1);
    if (trace) fclose(trace);
}

int InitializeInjection(void) {
    const char *path = getenv("H3_CUPTI_OUT");
    if (!path) return 0;
    trace = fopen(path, "w");
    if (!trace) return 0;
    uint64_t cupti_now = 0;
    struct timespec now;
    cuptiGetTimestamp(&cupti_now);
    clock_gettime(CLOCK_MONOTONIC, &now);
    fprintf(trace, "C %llu %llu\n", (unsigned long long)cupti_now,
            (unsigned long long)now.tv_sec * 1000000000ull +
                (unsigned long long)now.tv_nsec);
    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    const char *api = getenv("H3_CUPTI_API");
    if (api && api[0] == '1') {
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    }
    atexit(finish);
    return 1;
}
