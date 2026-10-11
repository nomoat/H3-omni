#ifndef H3_FASTH3_H
#define H3_FASTH3_H
#include <stddef.h>
#include "h3_host.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Native exports are explicit: a VSA checkpoint must never silently run dense. */
enum { H3_FASTH3_NONE = 0, H3_FASTH3_DENSE = 1, H3_FASTH3_VSA = 2 };
typedef struct {
    int kind;
    int steps;
    int rungs[4];
    float video_shift, audio_shift;
} h3_fasth3_config;
/* Missing manifest means original H3. Malformed/unsupported manifests fail. */
int h3_fasth3_config_read(const char *transformer_dir, h3_fasth3_config *config,
                         char *error, size_t error_size);
int h3_fasth3_schedule_build(const h3_fasth3_config *config,
                            h3_sigma_schedule *schedule);
#ifdef __cplusplus
}
#endif
#endif
