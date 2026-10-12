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

The contract below also applies to incremental sessions. In `h3_generate`, all
audio guidance is prepared before student generation. In a session, guidance
is prepared only for the current request, with student KV retained while the
audio teacher runs.

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
For cross-call history, the low-level `h3_live_session` interface described below
keeps one resident DiT dedicated to a stream.
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

`tests/live/resident.sbatch` runs complete audio/video generation on the
requested number of H200s. Set `H3_LIVE_BENCHMARK=1` to add three warm
measurements each at 124 and 243 frames. It checks small and 864x480 canvases,
repeated resident requests, and GPU-versus-host sampling. All repetitions at
the same resolution and duration must match byte for byte.
Measurements and artifacts are recorded with the completed validation run.

### H200 result, 2026-10-12

At 864x480, 24 fps, 243 frames (10.125 seconds), including generated audio,
VAE decoding and MP4 encoding:

| Configuration | First full-size request after the small case | Warm GPU sampler | Warm host sampler |
| --- | --- | --- | --- |
| 1 H200 | 37.41 s | 21.27 s | 21.42 s |
| 2 H200, Ulysses | 28.46 s | 12.93 s | 13.19 s |
| 4 H200, Ulysses | 24.25 s | 8.38-8.87 s (three runs) | 8.60 s |

The six full-size MP4 files from one and two GPUs (cold/warm/host sampler)
are byte-identical. The small-case video/audio latents and decoded outputs also
match exactly between one and two GPUs. The sample was inspected around
chunk boundaries and the first request boundary; appearance stayed consistent.

These are individual measurements, not a latency distribution. The initial
small request includes model loading from shared storage and takes minutes.
Two GPUs are slower than playback. Four GPUs are faster than playback in
these short warm runs, but this does not validate a sustained live broadcast.
The ordinary `h3_generate` API still decodes after latent generation and resets
history between independent calls. It retains its 362-frame limit (three
requests). The incremental session below removes that limit for latent
generation; a production service still needs bounded decoding and transport.

The regression job passed 69 existing CUDA primitive cases, six rectangular
attention cases, eight released-adapter projection cases plus the FP32 sampler,
and 21 exact sharding checks. The three checkpoint modes load successfully,
and a mismatched `--mode` is rejected. See
[`tests/live/evidence`](../tests/live/evidence/) for measurements and checks.

### Four-GPU follow-up (job 15202)

The runtime is unchanged from `1a1f588`. On `gpu-h200x8-0`, four H200s and
32 allocated CPU threads ran ten complete generations in one resident process.
The benchmark adds three warm runs at each duration:

| Frames / playback | Warm wall time | Median | Playback / generation speed |
| --- | --- | --- | --- |
| 124 / 5.167 s | 4.47-4.63 s | 4.52 s | 1.14x |
| 243 / 10.125 s | 8.38-8.87 s | 8.62 s | 1.17x |

Times include audio guidance, video generation, both VAEs and MP4 encoding.
The first small request takes 145.21 s including loading; the first full-size
10-second and 5-second cases take 24.25 s and 19.67 s. Warm speed requires a
resident, prepared process. Three runs on one prompt are not a long-running
stability or quality evaluation.

All repetitions within the four-GPU configuration match byte for byte,
including host-versus-GPU sampling, and all ten MP4 files decode successfully.
Four-GPU outputs do **not** match the one/two-GPU outputs byte for byte.
An isolated projection probe reproduces shape-dependent cuBLAS accumulation
changes in the base MLP FC2 when splitting the small sequence four ways;
attention split from 56 to 14 heads remains exact. The small-case final audio
and video latent relative L2 differences are 0.232 and 0.272. End-to-end
amplification cannot be judged by the small local projection error alone.
This follow-up records the difference without changing the runtime. Strict
cross-GPU-count parity for four GPUs remains open. Full-size frame inspection
around chunk and request boundaries shows consistent appearance in this sample.

Evidence: [`tests/live/evidence/15202`](../tests/live/evidence/15202/).
To repeat using the documented build cache:

```sh
sbatch --export=ALL --nodelist=gpu-h200x8-0 tests/live/benchmark.sbatch \
  /home/cj/h3-live/build-cache-15182.tar /home/cj/h3-live/benchmark-4gpu.tar
```

## Incremental latent sessions

`h3_live.h` exposes `h3_live_session_new`, `h3_live_session_next`, and
`h3_live_session_free`. A session borrows a live DiT and a text embedding;
these must stay alive and cannot serve another request during the session.
Creation resets history once. Each successful `next` returns one new request:
124 frames / 37 video latents first, then 119 frames / 35 video latents.
Audio counts follow the cumulative 40 Hz timeline. Output arrays are
channel-major FP32; allocate at least `24*37*latent_height*latent_width`
video floats and `64*207` audio floats.

The session retains historical KV in the DiT, first-chunk normalization
statistics, the previous 40 audio latents, and absolute timeline positions.
Each call generates its own audio guidance and four student chunks, committing
clean KV after each chunk. No later request is generated early. An inference
failure makes the session terminal; freeing it does not free the borrowed DiT
or embedding. The interface currently keeps one fixed prompt and canvas.

`tests/live/continuity.c` exercises ten sequential requests on one resident
model and emits each part before calling `next` again. It first compares the
first two requests against the existing 243-frame path, element for element.
For decoder correctness it decodes the known latent prefix and holds back
24 frames (one second). At every request it compares previously emitted RGB
and PCM against the new prefix decode. This repeated prefix decode is an
explicit test oracle, not a bounded-cost streaming decoder or latency claim.
The emitted parts contain 100, eight times 119, and 143 frames. Together they
contain 1,195 frames (49.7917 seconds). The combined file encodes their original
RGB/PCM once, avoiding independent AAC encoder padding at concatenation seams.

Reproduce on four H200s with the existing build cache:

```sh
sbatch --export=ALL tests/live/continuity.sbatch \
  /home/cj/h3-live/build-cache-15182.tar /home/cj/h3-live/continuity-overlay.tar
```

The overlay must contain `h3_live.c`, `h3_live.h`, `cuda/Makefile`, and the three
`tests/live/continuity*` files from this revision. The script retains the cached
build path because compiled object dependencies and ICU RPATHs refer to it.

The first ten-request run passed on four H200s (job 15225). Its first two
requests exactly match the legacy 10-second path. Across all ten requests,
previously emitted RGB remained byte-identical to later prefix decodes;
PCM changed by at most 3.07e-6. Sampled inspection of all nine request seams
and nine output seams found no obvious identity or background switch.
See [the session evidence](../tests/live/evidence/15225/README.md) for the
measurements, inspection scope and limitations. This validates one fixed
prompt and seed over 49.79 seconds, not a long-running interactive broadcast.

## Finite HTTP live pilot

`h3_generate_live` uses the resident context, one prompt and 1–12 connected
requests. `serve/live_server.py` exposes exactly 12 requests at `POST /v1/live`
with the existing bearer-key scheme. `serve/h3-live.sbatch` starts the two-H200
pilot on port 18961. A separate endpoint/key keeps it independent of FastH3.
The response is a growing fragmented MP4 with H.264 constrained baseline 3.1
and AAC-LC. One FFmpeg process spans all parts, with keyframes every second.
Clients can append bytes to a browser MediaSource before generation completes.

The fixed public session produces 1,433 frames (59.7083 seconds). The decoder
retains the tested one-second output holdback and repeated-prefix decode;
its maximum memory and work are bounded by the 12-request cap. No sustained
real-time claim is made. The service admits one active session, rejects a
second with 429, and notices disconnects while waiting or writing. It requests
cancellation through a private file; the worker stops at the next phase
boundary and resets KV for the next independent session. Resident weights stay.

CPU checks: `tests/live/stream_encoder.c` proves media exists before encoder
finish; `python3 -m unittest discover -s tests/live -p test_live_server.py -v`
checks incremental delivery, admission, limits and disconnect cancellation.

The final native audio endpoint can be slightly shorter than the video
endpoint because audio latents span 800 samples. Twelve requests produce
1,910,400 native audio samples, 267 samples (8.34 ms) short of the rounded
video endpoint. The adapter ends audio at the native boundary; intermediate
parts still use absolute video timestamps. It never pads each part or reads
beyond the decoded PCM. The CPU regression covers every 1–12 request endpoint
and cancellation using stub model/decoder output and the real adapter:

```sh
cc -std=c11 -g -fsanitize=address,undefined -I. \
  tests/live/stream_lengths.c h3_live_stream.c -lm -o /tmp/h3-live-lengths-test
/tmp/h3-live-lengths-test
```
