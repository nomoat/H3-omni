/* CPU-only: fragmented output must contain media before the encoder closes. */
#include "h3_ffmpeg.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc,char **argv) {
    assert(argc==2);
    char error[512]="";
    unsigned char *rgb=malloc(64*64*3*48);
    float *pcm=malloc(2*64000*sizeof(float));
    assert(rgb && pcm);
    for(int i=0;i<2*64000;i++) pcm[i]=0.1f*sinf((i%64000)*0.04f);
    h3_ffmpeg_live *s=h3_ffmpeg_live_open(argv[1],64,64,error,sizeof(error));
    assert(s);
    for(int p=0;p<3;p++) {
        memset(rgb,40+p*60,64*64*3*48);
        assert(h3_ffmpeg_live_push(s,rgb,48,pcm,64000,error,sizeof(error)));
    }
    int found=0;
    for(int attempt=0;attempt<50 && !found;attempt++) {
        FILE *f=fopen(argv[1],"rb");
        if(f) {
            char buf[65536]; size_t n=fread(buf,1,sizeof(buf),f); fclose(f);
            for(size_t i=0;i+4<=n;i++) if(!memcmp(buf+i,"moof",4)) found=1;
        }
        if(!found) usleep(20000);
    }
    assert(found);
    assert(h3_ffmpeg_live_finish(s,error,sizeof(error)));
    free(rgb); free(pcm); puts("fragmented media available before encoder finish");
    return 0;
}
