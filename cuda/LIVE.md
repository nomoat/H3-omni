# TaoMate-H3 live mode

The third checkpoint mode is `live`. It uses the original MiniMax-H3 FL2VA
weights plus the released TaoMate-H3 rank-128 adapter. `original` and `fast`
keep their existing weights and sampling contracts.

Sources: [TaoMate-H3 code](https://github.com/TaoLiveAIGC/TaoMate-H3/tree/6b2f9984acf681e2f387188f5d1f3b26e504315b)
and [released adapter](https://huggingface.co/TaoLiveAIGC/TaoMate-H3).
The adapter SHA-256 is
`a3ede4a1dce7030adc738fd8926ffd090e1b92e7f38d569626a35a553c0409ad`.

## Prepare and run

Download `adapter_model.safetensors`, `config.json`, and
`adapter_config.json` from the adapter repository. Convert once on a CPU host:

```sh
python3 cuda/prepare_live.py --base /models/MiniMax-H3 \
  --adapter /models/TaoMate-H3 --output /models/h3-live
make cuda CUDA_ARCH=sm_90

build-cuda/h3 --mode live -d /models/h3-live --gpus 1 \
  -p 'A presenter talks to the camera in a bright studio.' \
  --width 864 --height 480 --frames 243 --seed 42 -o live-1gpu.mp4

build-cuda/h3 --mode live -d /models/h3-live --gpus 2 \
  -p 'A presenter talks to the camera in a bright studio.' \
  --width 864 --height 480 --frames 243 --seed 42 -o live-2gpu.mp4
```

The export links to the base weights and writes only the BF16 adapter buffers.
Do not move the base directory afterwards. The runtime performs the two LoRA
matrix products separately; it does not merge LoRA into rounded base weights.
QKV adapter output rows are converted to the native interleaved head layout.
No Python or PyTorch is needed at inference time.

`--mode` checks the checkpoint type; the manifest also selects it automatically.
Live mode defaults to `--steps 3` and rejects a different step count. Use
124, 243, or 362 frames for one, two, or three complete requests. At 24 fps,
243 frames play for 10.125 seconds. A continuation does not repeat the initial
five-frame VAE prefix. Image/reference conditioning and denoising previews are
not part of this mode.

## Execution contract

1. The original base model, with the adapter disabled, generates audio guidance
   in nine evaluations per request. Later requests use the previous 40 clean
   audio latents per channel as a frozen reference. States 3, 6 and 9 guide
   the corresponding student steps.
2. Each request is split into four chunks. The first request produces
   39/34/34/17 frames; continuations produce 34/34/34/17. Audio boundaries use
   the cumulative timeline, rather than rounding each chunk independently.
3. Each chunk uses three noisy evaluations at retained states 0/16/33 of the
   shifted 50-state grid, followed by one clean evaluation at state 49. The
   video/audio shifts are 12/3. Video latents stay on the GPU during the three
   updates; audio is replaced with the corresponding teacher state.
4. Clean video statistics are matched to the first chunk before writing history.
   All 50 layers stage their clean media K/V before publishing a cache commit.
   Text attends only to current text. Current media attends to current text,
   historical media and current media. Each GPU stores its own attention heads.
5. Cache retention keeps the first chunk's video plus the latest two complete
   audio/video chunks. Current text is never stored. A new generation clears
   history. Absolute media positions continue across chunks and requests.
6. The assembled latent sequence uses the existing video/audio VAEs and MP4
   writer. The resident API also reuses concurrent VAE decoding and MP4 encoding.

Shared optimizations include resident weights, three cached AdaLN tables,
cuBLAS projections, fixed 512-row LoRA down projections, fused
normalization/RoPE/residual operations, dense
FlashAttention-3, Ulysses/NCCL and FP16 video VAE tiles. The LoRA branch must run
before QK normalization and SwiGLU. FastH3's VSA weights and sparse attention
are specific to that separately trained model and are not applied here.

The existing HTTP worker can use this export with `serve/server.py --steps 3`
and a live model directory. Its requests are still independent generations.

This implements the released H3 chunk-generation pipeline. Audio guidance is
prepared first, and decoding starts after latent generation. It does not add
an RTMP/WebRTC server or an interactive prompt queue. Independent `h3_generate`
calls start independent streams; history carries between chunks inside a call.
The native normal RNG differs from `torch.randn`, and the normalization reducer
also has a different floating-point reduction order. Equal seeds do not imply
bit-identical output with the upstream Python program.

## Validation

```sh
cc -O1 -std=c11 -D_GNU_SOURCE -DH3_CUDA -ffp-contract=off \
  -ffunction-sections -fdata-sections -I. tests/live/host.c h3_host.c h3_dit.c \
  -Wl,--gc-sections -lm -o /tmp/h3-live-host-test
/tmp/h3-live-host-test
python3 tests/live/attention.py --output attention.json
python3 tests/live/lora.py --output lora.json
python3 tests/live/sharding.py
```

The host test checks the global timeline, frozen audio reference, sampler and
video normalization. GPU tests compare rectangular attention with independent
PyTorch math, compare native LoRA projections with the original adapter, and
check GPU sampler operation order against FP32 host arithmetic. The sharding
test requires exact equality after splitting token rows or attention heads.
LoRA down projections use a fixed row tile because cuBLAS otherwise changes
its accumulation order with the per-GPU row count; this small local change can
produce a large change after the distilled model's three steps.

`tests/live/resident.sbatch` runs complete audio/video generation on one or two
H200s. It checks small and 864x480 canvases, repeated resident requests, and
GPU-versus-host sampling. The last three MP4 files must match byte for byte.
Measurements and artifacts are recorded with the completed validation run.

### H200 result, 2026-10-12

At 864x480, 24 fps, 243 frames (10.125 seconds), including generated audio,
VAE decoding and MP4 encoding:

| Configuration | First full-size request after the small case | Warm GPU sampler | Warm host sampler |
| --- | --- | --- | --- |
| 1 H200 | 37.41 s | 21.27 s | 21.42 s |
| 2 H200, Ulysses | 28.46 s | 12.93 s | 13.19 s |

All six full-size MP4 files (cold/warm/host sampler on each GPU count) are
byte-identical. The small-case video/audio latents and decoded outputs also
match exactly between one and two GPUs. The sample was inspected around
chunk boundaries and the first request boundary; appearance stayed consistent.

These are individual measurements, not a latency distribution. The initial
small request includes model loading from shared storage and takes minutes.
The two-GPU warm result is still slower than playback. This release validates
chunk generation and continuity; it does not yet provide a real-time broadcast
service. The current API retains its 362-frame limit (three requests).

The regression job passed 69 existing CUDA primitive cases, six rectangular
attention cases, eight released-adapter projection cases plus the FP32 sampler,
and 21 exact sharding checks. The three checkpoint modes load successfully,
and a mismatched `--mode` is rejected. See
[`tests/live/evidence`](../tests/live/evidence/) for measurements and checks.
