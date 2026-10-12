/* Validate timeline, packing, sampler and normalization without a GPU. */
#include <assert.h>
#include "../../h3_live.c"
int main(void) {
    h3_sigma_schedule s;
    assert(h3_live_schedule_build(&s) && s.steps == 4);
    assert(s.video[0] == 1 && s.video[3] == 0 && s.video[4] == 0);
    assert(s.audio[0] == 1 && s.audio[3] == 0);
    for (int i=0;i<3;i++) assert(s.video[i] > s.video[i+1]);
    assert(audio_boundary(39)==65 && audio_boundary(124)==207);
    assert(audio_boundary(243)==405 && audio_boundary(362)==603);
    assert(video_time(0)==0 && video_time(5)==85.0/3.0);
    h3_text_embedding text = {.tokens=7, .width=5120};
    h3_layout l={0}; char e[256]={0};
    assert(layout_build(&l,&text,10,16,16,57,37,37,207,0,0,e,sizeof(e)));
    assert(l.signature[1]==10 && l.img_target_rows==640 && l.audio_target_rows==114);
    assert(l.positions[0].t==video_time(37));
    assert(l.positions[7].t==214);
    assert(l.positions[7+57].t==214);
    assert(l.positions[7+114].t==7+video_time(37));
    h3_layout_free(&l);
    assert(layout_build(&l,&text,0,16,16,198,0,0,0,1,207,e,sizeof(e)));
    assert(l.signature[1]==0 && l.img_target_rows==0 && l.audio_cond_rows==80);
    assert(l.seq_len==7+80+396);
    assert(l.positions[7].t==174 && l.positions[87].t==214);
    h3_layout_free(&l);
    float value=2, velocity=3;
    update(&value,&velocity,1,1,0); assert(value==5);
    float x[24*2*4*4], mean[96], std[96];
    for (int i=0;i<24*2*4*4;i++) x[i]=(i%31)*0.1f;
    assert(renorm(x,2,4,4,mean,std,1));
    for (int i=0;i<24*2*4*4;i++) x[i]=x[i]*2+4;
    assert(renorm(x,2,4,4,mean,std,0));
    for (int i=0;i<24*2*4*4;i++) assert(fabsf(x[i]-(i%31)*0.1f)<2e-6f);
    puts("live host contract passed");
    return 0;
}
