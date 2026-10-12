/* CPU regression for native audio/video rounding at every finite endpoint.
 * Stub the expensive model and decoder; run the real streaming adapter. */
#include "h3_live.h"
#include "h3_video_vae.h"
#include "h3_audio_vae.h"
#include "h3_ffmpeg.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_live_session { int request; };
struct h3_ffmpeg_live { int unused; };
static int frames_written, samples_written, pushes, finishes, aborts, cancel_at;

h3_live_session *h3_live_session_new(h3_dit *d,const h3_text_embedding *t,
    int h,int w,uint64_t seed,char *e,size_t n) {
    assert(h==1 && w==1); return calloc(1,sizeof(h3_live_session));
}
int h3_live_session_next(h3_live_session *s,float *v,size_t vc,float *a,size_t ac,
    h3_live_part *p,h3_dit_progress cb,void *o,char *e,size_t n) {
    int old_frames=s->request?5+119*s->request:0;
    int old_audio=(old_frames*5+1)/3;
    int old_video=s->request?2+35*s->request:0;
    s->request++;
    int end=5+119*s->request;
    *p=(h3_live_part){s->request-1,old_frames,end-old_frames,
        old_video,2+35*s->request-old_video,old_audio,(end*5+1)/3-old_audio};
    memset(v,0,vc*sizeof(float)); memset(a,0,ac*sizeof(float)); return 1;
}
void h3_live_session_free(h3_live_session *s) { free(s); }
int h3_video_vae_decoder_decode_rgb8(h3_video_vae_decoder *d,const float *l,
    int t,h3_video_frames *out,int keep,uint8_t **rgb,h3_video_frames_ready cb,
    void *o,char *e,size_t n) {
    out->frames=5+119*((t-2)/35);
    *rgb=calloc((size_t)out->frames,16*16*3); return *rgb!=NULL;
}
int h3_audio_vae_decoder_decode(h3_audio_vae_decoder *d,const float *l,int t,
    h3_audio_vae_progress cb,void *o,h3_audio_waveform *out,char *e,size_t n) {
    out->samples=t*800;
    out->pcm=calloc((size_t)out->samples*2,sizeof(float)); return out->pcm!=NULL;
}
void h3_video_frames_free(h3_video_frames *f) { memset(f,0,sizeof(*f)); }
void h3_audio_waveform_free(h3_audio_waveform *w) { free(w->pcm); memset(w,0,sizeof(*w)); }
h3_ffmpeg_live *h3_ffmpeg_live_open(const char *p,int w,int h,char *e,size_t n) {
    return calloc(1,sizeof(h3_ffmpeg_live));
}
int h3_ffmpeg_live_push(h3_ffmpeg_live *s,const uint8_t *rgb,int f,
    const float *pcm,int a,char *e,size_t n) {
    assert(f>0 && a>0);
    /* With ASan, these accesses also verify the actual last output buffers. */
    assert(rgb[(size_t)f*16*16*3-1]==0 && pcm[(size_t)a*2-1]==0);
    frames_written+=f; samples_written+=a; pushes++; return 1;
}
int h3_ffmpeg_live_finish(h3_ffmpeg_live *s,char *e,size_t n) { finishes++; free(s); return 1; }
void h3_ffmpeg_live_abort(h3_ffmpeg_live *s) { if(s) { aborts++; free(s); } }
static int progress(const char *phase,int completed,int total,void *opaque) {
    return cancel_at && !strcmp(phase,"live generating") && completed==cancel_at;
}
int main(void) {
    h3_params p={0}; p.width=p.height=16; p.output_path="stub"; p.on_progress=progress;
    for(int requests=1;requests<=12;requests++) {
        frames_written=samples_written=pushes=finishes=aborts=0;
        char error[512]="";
        assert(h3_live_stream_generate(NULL,NULL,NULL,NULL,&p,requests,error,sizeof(error)));
        int frames=5+119*requests;
        int target=(int)llround((double)frames*32000/24);
        int native=((frames*5+1)/3)*800;
        assert(frames_written==frames && samples_written==(target<native?target:native));
        assert(pushes==requests && finishes==1 && aborts==0);
    }
    frames_written=samples_written=pushes=finishes=aborts=0; cancel_at=3;
    char error[512]="";
    assert(!h3_live_stream_generate(NULL,NULL,NULL,NULL,&p,12,error,sizeof(error)));
    assert(!strcmp(error,"generation cancelled"));
    assert(pushes==3 && finishes==0 && aborts==1);
    puts("all 12 native endpoints and cancellation passed");
}
