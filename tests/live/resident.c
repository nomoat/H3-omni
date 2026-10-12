/* Full live-mode validation in the same resident API used by the server. */
#include "h3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char **argv) {
    if (argc!=3) return 2;
    h3_ctx *ctx=h3_load_dir(argv[1]);
    if (!ctx) { fprintf(stderr,"%s\n",h3_last_error(NULL)); return 1; }
    if (strcmp(h3_model_mode(ctx),"live") || h3_model_default_steps(ctx)!=3) return 1;
    h3_cache_set_enabled(ctx,1); h3_cache_set_resident_weights(ctx,1);
    const char *prompt="A woman with brown hair sits at a desk in a bright studio. She looks at the camera, smiles and says, \"Welcome to the show. Today we will explore something new together.\" A steady medium shot.";
    const char *names[]={"small", "full-cold", "full-warm", "full-host"};
    int ok=1;
    for (int i=0;i<4;i++) {
        char output[2048],dump[2048];
        snprintf(output,sizeof(output),"%s/%s.mp4",argv[2],names[i]);
        snprintf(dump,sizeof(dump),"%s/%s-latents",argv[2],names[i]);
        if (i==0) { mkdir(dump,0755); setenv("H3_DUMP_DIR",dump,1); }
        else unsetenv("H3_DUMP_DIR");
        if (i==3) setenv("H3_LIVE_HOST_SAMPLER","1",1);
        else unsetenv("H3_LIVE_HOST_SAMPLER");
        h3_params p=H3_PARAMS_DEFAULT;
        p.width=i?864:256; p.height=i?480:256; p.frames=243; p.steps=3;
        p.seed=42; p.output_path=output;
        double started=now();
        h3_result *r=h3_generate(ctx,prompt,&p);
        if (!r) { fprintf(stderr,"case %s: %s\n",names[i],h3_last_error(ctx)); ok=0; break; }
        h3_result_free(r);
        printf("{\"case\":\"%s\",\"width\":%d,\"height\":%d,\"frames\":243,\"seconds\":%.6f}\n",
               names[i],p.width,p.height,now()-started); fflush(stdout);
    }
    h3_free(ctx);
    return ok?0:1;
}
