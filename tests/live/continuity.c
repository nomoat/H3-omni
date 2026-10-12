/* Correctness-first incremental session test. Decode the known prefix as an
 * oracle, retain one second of lookahead, and publish BEFORE requesting more.
 * This deliberately does not benchmark a bounded streaming VAE implementation. */
#include "h3_internal.h"
#include "h3_live.h"
#include "h3_video_vae.h"
#include "h3_audio_vae.h"
#include "h3_ffmpeg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s: %s\n",__LINE__,#x,error); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void append(float *all,int total,const float *part,int n,int start,int channels,size_t area) {
    for(int c=0;c<channels;c++) memcpy(all+((size_t)c*total+start)*area,
        part+(size_t)c*n*area,(size_t)n*area*sizeof(float));
}
static void prefix(float *dst,const float *src,int total,int n,int channels,size_t area) {
    for(int c=0;c<channels;c++) memcpy(dst+(size_t)c*n*area,
        src+(size_t)c*total*area,(size_t)n*area*sizeof(float));
}
static int compare_file(const char *dir,const char *name,const float *a,size_t n) {
    char path[4096]; snprintf(path,sizeof(path),"%s/%s.f32",dir,name);
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    float *b=malloc(n*sizeof(float)); if(!b) { fclose(f); return 0; }
    int ok=fread(b,sizeof(float),n,f)==n && fgetc(f)==EOF;
    size_t different=0; double ss=0,rr=0,maximum=0;
    if(ok) for(size_t i=0;i<n;i++) {
        if(memcmp(a+i,b+i,sizeof(float))) different++;
        double d=(double)a[i]-b[i]; ss+=d*d; rr+=(double)b[i]*b[i];
        if(fabs(d)>maximum) maximum=fabs(d);
    }
    printf("{\"check\":\"legacy_%s\",\"different\":%zu,\"count\":%zu,\"max_abs\":%.9g,\"relative_l2\":%.9g}\n",name,different,n,maximum,sqrt(ss/(rr+1e-30)));
    free(b); fclose(f); return ok && !different;
}
static void save(const char *dir,const char *name,const void *p,size_t bytes) {
    char error[512]="",path[4096]; snprintf(path,sizeof(path),"%s/%s",dir,name);
    FILE *f=fopen(path,"wb"); CHECK(f); CHECK(fwrite(p,1,bytes,f)==bytes); CHECK(!fclose(f));
}
int main(int argc,char **argv) {
    if(argc!=4) { fprintf(stderr,"usage: %s MODEL OUTPUT LOCAL_SCRATCH\n",argv[0]); return 2; }
    enum {N=10,W=864,H=480,FPS=24,SR=32000,HOLD=24};
    int final_frames=5+119*N, final_vt=2+35*N, final_at=(final_frames*5+1)/3;
    int final_samples=(int)llround((double)final_frames*SR/FPS);
    size_t area=(W/16)*(H/16),frame_bytes=(size_t)W*H*3;
    char error[512]="",path[4096];
    h3_ctx *ctx=h3_load_dir(argv[1]); CHECK(ctx);
    CHECK(!strcmp(h3_model_mode(ctx),"live"));
    h3_cache_set_enabled(ctx,1); h3_cache_set_resident_weights(ctx,1);
    const char *prompt="A woman with brown hair sits at a desk in a bright studio. She looks at the camera, smiles and says, \"Welcome to the show. Today we will explore something new together.\" A steady medium shot.";
    mkdir(argv[3],0755); setenv("H3_DUMP_DIR",argv[3],1);
    snprintf(path,sizeof(path),"%s/legacy-10s.mp4",argv[2]);
    h3_params params=H3_PARAMS_DEFAULT;
    params.width=W; params.height=H; params.frames=243; params.steps=3; params.seed=42; params.output_path=path;
    h3_result *baseline=h3_generate(ctx,prompt,&params);
    if(!baseline) snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
    CHECK(baseline); h3_result_free(baseline); unsetenv("H3_DUMP_DIR");
    h3_text_embedding text={.tokens=ctx->conditioning_tokens,.width=ctx->conditioning_width,
        .values=ctx->conditioning_values,.tags=ctx->conditioning_tags};
    CHECK(ctx->dit && ctx->video_decoder && ctx->audio_decoder && text.values);
    h3_live_session *session=h3_live_session_new(ctx->dit,&text,H/16,W/16,42,error,sizeof(error)); CHECK(session);
    float *v=malloc(24*37*area*sizeof(float)),*a=malloc(64*207*sizeof(float));
    float *allv=calloc(24*final_vt*area,sizeof(float)),*alla=calloc(64*final_at,sizeof(float));
    float *pv=malloc(24*final_vt*area*sizeof(float)),*pa=malloc(64*final_at*sizeof(float));
    uint8_t *published=malloc((size_t)final_frames*frame_bytes);
    float *pcm=calloc((size_t)2*final_samples,sizeof(float));
    CHECK(v && a && allv && alla && pv && pa && published && pcm);
    int emitted_frames=0,emitted_samples=0,parity=0;
    size_t total_changed=0; double max_audio=0;
    for(int r=0;r<N;r++) {
        h3_live_part part; double begin=now();
        CHECK(h3_live_session_next(session,v,24*37*area,a,64*207,&part,NULL,NULL,error,sizeof(error)));
        double generated=now();
        int frames=part.frame_start+part.frame_count,vt=part.video_start+part.video_t,at=part.audio_start+part.audio_t;
        CHECK(part.request_index==r && part.frame_start==(r?5+119*r:0));
        append(allv,final_vt,v,part.video_t,part.video_start,24,area);
        append(alla,final_at,a,part.audio_t,part.audio_start,64,1);
        prefix(pv,allv,final_vt,vt,24,area); prefix(pa,alla,final_at,at,64,1);
        if(r==1) {
            int vok=compare_file(argv[3],"video_latent",pv,24*vt*area);
            int aok=compare_file(argv[3],"audio_latent",pa,64*at);
            parity=vok && aok; fflush(stdout); CHECK(parity);
        }
        h3_video_frames decoded={0}; uint8_t *rgb=NULL; h3_audio_waveform wave={0};
        CHECK(h3_video_vae_decoder_decode_rgb8(ctx->video_decoder,pv,vt,&decoded,0,&rgb,NULL,NULL,error,sizeof(error)));
        CHECK(decoded.frames==frames && decoded.width==W && decoded.height==H);
        CHECK(h3_audio_vae_decoder_decode(ctx->audio_decoder,pa,at,NULL,NULL,&wave,error,sizeof(error)));
        CHECK(wave.channels==2 && wave.sample_rate==SR);
        size_t changed=0; int max_pixel=0; double audio_max=0,audio_ss=0;
        for(size_t i=0;i<(size_t)emitted_frames*frame_bytes;i++) {
            int d=abs((int)published[i]-rgb[i]); if(d) changed++; if(d>max_pixel) max_pixel=d;
        }
        for(int c=0;c<2;c++) for(int i=0;i<emitted_samples;i++) {
            double d=(double)pcm[(size_t)c*final_samples+i]-wave.pcm[(size_t)c*wave.samples+i];
            CHECK(isfinite(d)); if(fabs(d)>audio_max) audio_max=fabs(d); audio_ss+=d*d;
        }
        total_changed+=changed; if(audio_max>max_audio) max_audio=audio_max;
        int end=r==N-1?frames:frames-HOLD;
        int sample_end=(int)llround((double)end*SR/FPS); CHECK(sample_end<=wave.samples);
        int count=end-emitted_frames, samples=sample_end-emitted_samples;
        memcpy(published+(size_t)emitted_frames*frame_bytes,rgb+(size_t)emitted_frames*frame_bytes,(size_t)count*frame_bytes);
        float *cut=malloc((size_t)2*samples*sizeof(float)); CHECK(cut);
        for(int c=0;c<2;c++) {
            memcpy(cut+(size_t)c*samples,wave.pcm+(size_t)c*wave.samples+emitted_samples,(size_t)samples*sizeof(float));
            memcpy(pcm+(size_t)c*final_samples+emitted_samples,cut+(size_t)c*samples,(size_t)samples*sizeof(float));
        }
        snprintf(path,sizeof(path),"%s/part-%02d.mp4",argv[2],r+1);
        CHECK(h3_ffmpeg_write_av_rgb24_f32(path,rgb+(size_t)emitted_frames*frame_bytes,count,W,H,FPS,cut,samples,2,SR,error,sizeof(error)));
        printf("{\"request\":%d,\"generated_start\":%d,\"generated_end\":%d,\"published_start\":%d,\"published_end\":%d,\"sample_start\":%d,\"sample_end\":%d,\"clean_commits\":%d,\"generation_seconds\":%.6f,\"total_seconds\":%.6f,\"past_rgb_changed\":%zu,\"past_rgb_max\":%d,\"past_pcm_max\":%.9g,\"past_pcm_rms\":%.9g}\n",
            r+1,part.frame_start,frames,emitted_frames,end,emitted_samples,sample_end,4*(r+1),generated-begin,now()-begin,changed,max_pixel,audio_max,sqrt(audio_ss/(2.0*emitted_samples+1)));
        fflush(stdout); emitted_frames=end; emitted_samples=sample_end;
        free(cut); free(rgb); h3_video_frames_free(&decoded); h3_audio_waveform_free(&wave);
    }
    CHECK(emitted_frames==final_frames && emitted_samples==final_samples);
    snprintf(path,sizeof(path),"%s/continuous-50s.mp4",argv[2]);
    CHECK(h3_ffmpeg_write_av_rgb24_f32(path,published,final_frames,W,H,FPS,pcm,final_samples,2,SR,error,sizeof(error)));
    /* Raw output lives on node-local disk for seam analysis; copy only metrics. */
    save(argv[3],"published.rgb24",published,(size_t)final_frames*frame_bytes);
    save(argv[3],"published-pcm.f32",pcm,(size_t)2*final_samples*sizeof(float));
    save(argv[2],"session-video-latent.f32",allv,24*final_vt*area*sizeof(float));
    save(argv[2],"session-audio-latent.f32",alla,64*final_at*sizeof(float));
    printf("{\"summary\":true,\"requests\":%d,\"frames\":%d,\"seconds\":%.9g,\"samples\":%d,\"legacy_exact\":%s,\"total_past_rgb_changed\":%zu,\"max_past_pcm_change\":%.9g,\"hold_frames\":%d}\n",N,final_frames,(double)final_frames/FPS,final_samples,parity?"true":"false",total_changed,max_audio,HOLD);
    fflush(stdout);
    free(v); free(a); free(allv); free(alla); free(pv); free(pa); free(published); free(pcm);
    h3_live_session_free(session); h3_free(ctx);
    return total_changed || max_audio>1e-4 ? 1 : 0;
}
