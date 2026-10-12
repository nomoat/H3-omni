/* Private DiT live-mode operations. Included after the weight helpers. */
static int live_lora(h3_dit *d, const h3_dit_block *b, int projection,
                     h3_gpu_tensor *out, const h3_gpu_tensor *input,
                     uint32_t rows, uint32_t in, uint32_t width) {
    if (!d->live_adapter) return 1;
    if (d->lora_scratch_rows < rows) {
        free_tensor(&d->lora_mid);
        free_tensor(&d->lora_update);
        d->lora_mid = h3_gpu_tensor_new_bf16(d->gpu, (size_t)rows * 128);
        d->lora_update = h3_gpu_tensor_new_bf16(d->gpu, (size_t)rows * FFN * 2);
        d->lora_scratch_rows = d->lora_mid && d->lora_update ? rows : 0;
    }
    return d->lora_scratch_rows && b->lora_a[projection] && b->lora_b[projection] &&
#ifdef H3_CUDA
        h3_gpu_lora_down_bf16(d->gpu, d->lora_mid, input, b->lora_a[projection], rows, in) &&
#else
        h3_gpu_linear_bf16(d->gpu, d->lora_mid, input, b->lora_a[projection],
                           NULL, rows, in, 128) &&
#endif
        h3_gpu_linear_bf16(d->gpu, d->lora_update, d->lora_mid, b->lora_b[projection],
                           NULL, rows, 128, width) &&
        h3_gpu_add_bf16(d->gpu, out, out, d->lora_update, rows * width);
}

static void live_pair_free(h3_live_kv *p) {
    free_tensor(&p->key);
    free_tensor(&p->value);
    memset(p, 0, sizeof(*p));
}

static void live_history_free(h3_dit *d) {
    for (unsigned i = 0; i < H3_DIT_BLOCKS; i++) {
        live_pair_free(&d->live_sink[i]);
        live_pair_free(&d->live_recent[0][i]);
        live_pair_free(&d->live_recent[1][i]);
        live_pair_free(&d->live_pending[i]);
    }
    d->live_commits = 0;
}

#ifdef H3_CUDA
static int live_pair_copy(h3_dit *d, h3_live_kv *p, uint32_t start,
                          uint32_t rows, uint32_t audio, uint32_t width) {
    live_pair_free(p);
    p->key = h3_gpu_tensor_new_bf16(d->gpu, (size_t)rows * width);
    p->value = h3_gpu_tensor_new_bf16(d->gpu, (size_t)rows * width);
    p->rows = rows;
    p->audio = audio;
    return p->key && p->value &&
        h3_gpu_copy_bf16(d->gpu, p->key, 0, d->key, (size_t)start * width,
                          (size_t)rows * width) &&
        h3_gpu_copy_bf16(d->gpu, p->value, 0, d->value, (size_t)start * width,
                          (size_t)rows * width);
}

static int live_attention(h3_dit *d, unsigned layer, char *error, size_t capacity) {
    unsigned world = d->world > 1 ? (unsigned)d->world : 1;
    uint32_t heads = HEADS / world, width = heads * HEAD_DIM;
    uint32_t text = d->text_rows, sequence = d->sequence;
    if (world > 1) {
        h3_gpu_tensor *inputs[] = {d->query, d->key, d->value};
        for (int i = 0; i < 3; i++)
            if (!h3_gpu_ulysses_scatter_heads_bf16(d->gpu, d->comm,
                    inputs[i], d->qkv, d->shard_rows, HEADS, HEAD_DIM)) goto failed;
    }
    h3_live_kv *history[3];
    int count = 0;
    if (d->live_commits > 2) history[count++] = &d->live_sink[layer];
    if (d->live_commits > 1) history[count++] = &d->live_recent[0][layer];
    if (d->live_commits) history[count++] = &d->live_recent[1][layer];
    size_t total = sequence;
    for (int i = 0; i < count; i++) total += history[i]->rows - history[i]->skip;
    if (total * width > d->live_concat_elements) {
        free_tensor(&d->live_concat_k);
        free_tensor(&d->live_concat_v);
        d->live_concat_k = h3_gpu_tensor_new_bf16(d->gpu, total * width);
        d->live_concat_v = h3_gpu_tensor_new_bf16(d->gpu, total * width);
        d->live_concat_elements = d->live_concat_k && d->live_concat_v ? total * width : 0;
    }
    if (!d->live_concat_elements) goto failed;
    size_t offset = 0;
#define APPEND(k, v, start, rows) do { \
    size_t n = (size_t)(rows) * width; \
    if (n && (!h3_gpu_copy_bf16(d->gpu, d->live_concat_k, offset, k, (size_t)(start)*width, n) || \
              !h3_gpu_copy_bf16(d->gpu, d->live_concat_v, offset, v, (size_t)(start)*width, n))) goto failed; \
    offset += n; \
} while (0)
    APPEND(d->key, d->value, 0, text);
    for (int i = 0; i < count; i++) {
        h3_live_kv *p = history[i];
        APPEND(p->key, p->value, p->skip, p->rows - p->skip);
    }
    APPEND(d->key, d->value, text, sequence - text);
#undef APPEND
    if (!h3_gpu_cross_attention_bf16(d->gpu, d->attention_heads, 0,
            d->query, 0, d->key, d->value, 0, text, text, heads) ||
        !h3_gpu_cross_attention_bf16(d->gpu, d->attention_heads, text,
            d->query, text, d->live_concat_k, d->live_concat_v, 0,
            sequence - text, (uint32_t)total, heads)) goto failed;
    if (d->live_stage &&
        !live_pair_copy(d, &d->live_pending[layer], text, sequence - text,
                         d->audio_rows, width)) goto failed;
    if (world > 1 && !h3_gpu_ulysses_gather_heads_bf16(d->gpu, d->comm,
            d->attention_heads, d->query, d->shard_rows, HEADS, HEAD_DIM)) goto failed;
    return 1;
failed:
    fail(error, capacity, "live attention layer %u: %s", layer, h3_gpu_error(d->gpu));
    return 0;
}
#endif
