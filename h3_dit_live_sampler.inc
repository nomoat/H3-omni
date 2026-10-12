#ifdef H3_CUDA
/* Keep the current chunk on the GPU through all three noisy evaluations.
 * The audio teacher supplies each next-step state, without a device readback. */
static int live_chunk_device(h3_dit *d, float *video, float *audio,
                              const float *const milestones[3],
                              char *error, size_t capacity) {
    size_t vn = (size_t)d->video_rows*VIDEO_PATCH;
    size_t an = (size_t)d->audio_rows*AUDIO_CHANNELS;
    float *vr = malloc(vn*sizeof(float)), *ar = malloc(an*sizeof(float));
    int ok = vr && ar &&
        h3_dit_patchify_video(video,24,d->latent_t,d->latent_h,d->latent_w,vr,vn) &&
        h3_dit_pack_audio(audio,32,d->audio_t,ar,an) &&
        h3_gpu_tensor_write_f32_range(d->video_input,0,vr,vn) &&
        h3_gpu_tensor_write_f32_range(d->audio_input,0,ar,an);
    for (int i=0; ok && i<3; i++) {
        ok = encode_forward(d,i,1,0,0,error,capacity) &&
            h3_gpu_taomate_step_f32(d->gpu,d->video_input,d->video_output,vn,
                                     d->sigmas.video[i],d->sigmas.video[i+1]) &&
            h3_dit_pack_audio(milestones[i],32,d->audio_t,ar,an) &&
            h3_gpu_tensor_write_f32_range(d->audio_input,0,ar,an) && h3_gpu_submit(d->gpu);
    }
    if (ok) ok = h3_gpu_tensor_read_f32_range(d->video_input,0,vr,vn) &&
        h3_dit_unpatchify_video(vr,24,d->latent_t,d->latent_h,d->latent_w,video,vn);
    if (ok) memcpy(audio,milestones[2],an*sizeof(float));
    free(vr); free(ar);
    if (!ok && error && capacity && !*error)
        fail(error,capacity,"live GPU sampler: %s",h3_gpu_error(d->gpu));
    return ok;
}
#endif
