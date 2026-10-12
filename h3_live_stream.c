/* Finite streaming adapter. Prefix decoding is deliberately retained from the
 * continuity oracle. The 12-request cap bounds CPU memory and decoder work. */
#include "h3_live.h"
#include "h3_video_vae.h"
#include "h3_audio_vae.h"
#include "h3_ffmpeg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_time(float *dst, int dt, int start, const float *src,
                      int st, int n, int channels, size_t area) {
    for (int c=0;c<channels;c++)
        memcpy(dst+((size_t)c*dt+start)*area, src+(size_t)c*st*area,
               (size_t)n*area*sizeof(float));
}

int h3_live_stream_generate(h3_dit *dit, const h3_text_embedding *text,
    h3_video_vae_decoder *video_decoder, h3_audio_vae_decoder *audio_decoder,
    const h3_params *p, int requests, char *error, size_t capacity) {
    int final_frames=5+119*requests, final_vt=2+35*requests;
    int final_at=(final_frames*5+1)/3;
    int height=p->height/16, width=p->width/16;
    size_t area=(size_t)height*width, frame_bytes=(size_t)p->height*p->width*3;
    float *v=NULL,*a=NULL,*allv=NULL,*alla=NULL,*pv=NULL,*pa=NULL,*cut=NULL;
    uint8_t *rgb=NULL;
    h3_video_frames decoded={0}; h3_audio_waveform wave={0};
    h3_live_session *session=NULL; h3_ffmpeg_live *encoder=NULL;
    int ok=0, emitted_frames=0, emitted_samples=0;
#define REQUIRE(x) do { if (!(x)) goto done; } while (0)
#define PROGRESS(phase, n) do { \
    if (p->on_progress && p->on_progress(phase,n,requests,p->callback_opaque)) { \
        snprintf(error,capacity,"generation cancelled"); goto done; \
    } \
} while (0)
    session=h3_live_session_new(dit,text,height,width,p->seed,error,capacity);
    REQUIRE(session);
    v=malloc(24*37*area*sizeof(float)); a=malloc(64*207*sizeof(float));
    allv=malloc(24*final_vt*area*sizeof(float)); alla=malloc(64*final_at*sizeof(float));
    pv=malloc(24*final_vt*area*sizeof(float)); pa=malloc(64*final_at*sizeof(float));
    REQUIRE(v && a && allv && alla && pv && pa);
    encoder=h3_ffmpeg_live_open(p->output_path,p->width,p->height,error,capacity);
    REQUIRE(encoder);
    for (int r=0;r<requests;r++) {
        PROGRESS("live generating",r);
        h3_live_part part;
        REQUIRE(h3_live_session_next(session,v,24*37*area,a,64*207,&part,
            NULL,NULL,error,capacity));
        PROGRESS("live decoding",r);
        int frames=part.frame_start+part.frame_count;
        int vt=part.video_start+part.video_t,at=part.audio_start+part.audio_t;
        copy_time(allv,final_vt,part.video_start,v,part.video_t,part.video_t,24,area);
        copy_time(alla,final_at,part.audio_start,a,part.audio_t,part.audio_t,64,1);
        copy_time(pv,vt,0,allv,final_vt,vt,24,area);
        copy_time(pa,at,0,alla,final_at,at,64,1);
        REQUIRE(h3_video_vae_decoder_decode_rgb8(video_decoder,pv,vt,
            &decoded,0,&rgb,NULL,NULL,error,capacity));
        REQUIRE(h3_audio_vae_decoder_decode(audio_decoder,pa,at,NULL,NULL,
            &wave,error,capacity));
        PROGRESS("live encoding",r);
        int end=r==requests-1?frames:frames-24;
        int sample_end=(int)llround((double)end*32000/24);
        int samples=sample_end-emitted_samples;
        REQUIRE(decoded.frames==frames && wave.samples>=sample_end);
        cut=malloc((size_t)2*samples*sizeof(float)); REQUIRE(cut);
        for (int c=0;c<2;c++) memcpy(cut+(size_t)c*samples,
            wave.pcm+(size_t)c*wave.samples+emitted_samples,(size_t)samples*sizeof(float));
        REQUIRE(h3_ffmpeg_live_push(encoder,rgb+(size_t)emitted_frames*frame_bytes,
            end-emitted_frames,cut,samples,error,capacity));
        emitted_frames=end; emitted_samples=sample_end;
        free(cut); cut=NULL; free(rgb); rgb=NULL;
        h3_video_frames_free(&decoded); h3_audio_waveform_free(&wave);
        PROGRESS("live part",r+1);
    }
    ok=h3_ffmpeg_live_finish(encoder,error,capacity); encoder=NULL;
done:
    h3_ffmpeg_live_abort(encoder); h3_live_session_free(session);
    free(v); free(a); free(allv); free(alla); free(pv); free(pa); free(cut); free(rgb);
    h3_video_frames_free(&decoded); h3_audio_waveform_free(&wave);
    if (!ok && error && capacity && !*error)
        snprintf(error,capacity,"live stream allocation or decode failed");
    return ok;
#undef PROGRESS
#undef REQUIRE
}
