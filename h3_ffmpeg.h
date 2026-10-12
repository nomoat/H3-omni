#ifndef H3_FFMPEG_H
#define H3_FFMPEG_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_IMAGE_FIT_STRETCH = 0,
    H3_IMAGE_FIT_COVER = 1
} h3_image_fit;

/* Inspect the first visual stream without decoding it. H3_FFPROBE may select
 * an explicit ffprobe-compatible executable. */
int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size);

/* Decode one visual stream through FFmpeg. The caller owns channel-major F32
 * [3,height,width] RGB in [0,1]. */
int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size);

/* Decode a 24 fps visual stream to channel-major F32 [3,T,H,W] in [0,1].
 * The returned frame count is trimmed down to the released 5+17k cadence. */
int h3_ffmpeg_read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size);

/* Decode the first audio stream as channel-major stereo F32 at 32 kHz.
 * max_samples bounds allocation. truncate_at_limit is used for a video's
 * soundtrack; standalone clips report an error instead of silently trimming. */
int h3_ffmpeg_read_audio_f32(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size);

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size);

/* Encode RGB24 video and channel-major F32 PCM through two concurrent pipes.
 * No intermediate uncompressed media file is created. */
int h3_ffmpeg_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                 int frame_count, int width, int height,
                                 int fps, const float *pcm, int samples,
                                 int channels, int sample_rate,
                                 char *error, size_t error_size);

/* The same encoding with the video supplied in segments while FFmpeg runs.
 * open() starts FFmpeg in the background and streams the whole PCM, which
 * must stay valid until finish() or abort(); push() queues the next
 * bytes of RGB24 video (the caller keeps them valid until finish or abort)
 * and may be called from another thread; finish() waits for FFmpeg after all
 * frame_count frames were pushed. abort() stops FFmpeg and removes the file.
 * finish() and abort() free the stream. */
typedef struct h3_ffmpeg_av_stream h3_ffmpeg_av_stream;
h3_ffmpeg_av_stream *h3_ffmpeg_av_stream_open(
    const char *path, int frame_count, int width, int height, int fps,
    const float *pcm, int samples, int channels, int sample_rate,
    char *error, size_t error_size);
int h3_ffmpeg_av_stream_push(h3_ffmpeg_av_stream *stream,
                             const uint8_t *frames, size_t bytes);
/* For a stream opened with pcm NULL and samples 0: hand over the PCM
 * (valid until finish or abort). FFmpeg buffers the video until then. */
int h3_ffmpeg_av_stream_set_audio(h3_ffmpeg_av_stream *stream,
                                  const float *pcm, int samples);
int h3_ffmpeg_av_stream_finish(h3_ffmpeg_av_stream *stream,
                               char *error, size_t error_size);
void h3_ffmpeg_av_stream_abort(h3_ffmpeg_av_stream *stream);

/* Incremental A/V input, one encoder for the full session. Output is fragmented
 * MP4 (H.264 baseline level 3.1 / AAC-LC) and can be read before finish.
 * push consumes both buffers before returning. finish/abort free the handle. */
typedef struct h3_ffmpeg_live h3_ffmpeg_live;
h3_ffmpeg_live *h3_ffmpeg_live_open(const char *path, int width, int height,
                                   char *error, size_t error_size);
int h3_ffmpeg_live_push(h3_ffmpeg_live *stream, const uint8_t *rgb, int frames,
                        const float *pcm, int samples, char *error, size_t error_size);
int h3_ffmpeg_live_finish(h3_ffmpeg_live *stream, char *error, size_t error_size);
void h3_ffmpeg_live_abort(h3_ffmpeg_live *stream);

#endif
