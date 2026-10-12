# H3-omni

**Original H3, FastH3, and live generation in one C/CUDA inference engine.**

H3-omni builds on [antirez/h3.c](https://github.com/antirez/h3.c), Salvatore
Sanfilippo's native MiniMax-H3 implementation. We started with its C model code
and Apple Silicon implementation, then added a CUDA backend, multi-GPU execution,
FastH3 support, and stateful live generation. The work here extends that foundation.

The goal is straightforward: bring H3's different inference workflows into one
native codebase. Generate a complete video with the original model, use a
distilled checkpoint for faster clips, or carry history between chunks for a
live stream. The model logic, sampling, and session state live in C; GPU work
uses CUDA and native libraries. The inference executable needs no Python,
PyTorch, or libtorch runtime.

The maintained backend targets **Linux and NVIDIA H200**, on one node with
one or more GPUs. Recorded tests cover 1, 2, 4, and 8 H200s. The inherited Metal
backend remains in the tree; its documentation is in [METAL.md](METAL.md).

## Why this project exists

H3 now has several useful inference paths, each with its own weights, sampling
schedule, and attention behavior. H3-omni puts those paths behind the same
native engine, CLI, and C API. They share model loading, resident weights,
GPU primitives, video and audio decoders, and media output.

This is a practical implementation choice: a focused engine for the H3 family,
with explicit control over memory, GPU execution, and streaming state. It also
makes it possible to embed generation without a Python model framework. The
recorded benchmarks show that this approach can reach latency comparable to
the Python-based vLLM-Omni stack.

## Three inference modes

| Mode | Model | Use case | Sampling |
| --- | --- | --- | --- |
| `original` | Original MiniMax-H3 | Generate a complete video with audio | 20 denoising steps by default |
| `fast` | FastH3 Preview v1 VSA / Data-Free | Generate short clips with lower latency | Trained four-step schedule and video sparse attention |
| `live` | Original H3 plus the TaoMate-H3 adapter | Generate connected chunks while retaining history | Three student steps per chunk, audio guidance, and clean KV updates |

Each mode uses its own checkpoint contract. The model directory selects the
mode automatically; `--mode` can check that it matches your intent. FastH3 and
TaoMate-H3 use separately trained weights and schedules. Changing the original
model's step count does not select either one.

Live sessions retain historical attention keys and values (KV), audio context,
and timeline positions between requests. Each new session starts with empty
history. The included streaming service generates twelve connected parts from
one prompt, delivering about one minute of video as it becomes available.

Setup and model details: [FastH3](cuda/FASTH3.md) · [Live](cuda/LIVE.md).

## Measured performance

These are recorded H200 runs from October 11–12, 2026. Clip benchmarks use
**864 × 480, 24 fps, audio, and complete MP4 output**. Weights stay resident
between requests. A “5 s” request produces 124 frames (5.17 s); a “10 s” request
produces 243 frames (10.125 s).

Times below are seconds of wall time, not just denoising time. Ranges include
all recorded requests of that duration after the initial warmup. They cover
a few prompts and repeats on specific builds and nodes, not a broad benchmark
suite or a guarantee for every workload.

### Original H3: comparable to vLLM-Omni

Both implementations use the original model and 20 denoising steps. H3-omni
times include generation, decoding, and encoding; vLLM-Omni times also include
HTTP transport to the client. The measurements were collected in separate runs
on the same H200 cluster, with different software stacks and some different
nodes. This is a latency comparison, not a claim of identical output quality.

| H200 GPUs | H3-omni, 5 s | vLLM-Omni, 5 s | H3-omni, 10 s | vLLM-Omni, 10 s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.94–32.06 | 32.96–33.03 | 80.16–80.50 | 82.83–82.90 |
| 2 | 18.06–18.32 | 18.92–18.96 | 44.24–44.66 | 45.42–45.52 |
| 4 | 9.17–9.42 | 10.21–10.35 | 22.16–22.35 | 23.96–24.09 |
| 8 | 5.40–5.72 | 6.06–6.10 | 12.56–13.06 | 13.57–13.58 |

Native timing records: [1 GPU](tests/serving/evidence/14771-duration/events.jsonl),
[2 GPUs](tests/serving/evidence/14781-multigpu-2/events.jsonl),
[4 GPUs](tests/serving/evidence/14789-multigpu-4/events.jsonl),
[8 GPUs](tests/serving/evidence/14792-multigpu-8/events.jsonl).
The [vLLM-Omni report](cuda/VLLM_OMNI_BENCH.md) records its versions, launch flags,
request settings, and timings. That report also contains an earlier, slower
H3-omni baseline; the table above uses the later native measurements linked here.

### FastH3: a five-second request in 1.65–1.95 seconds on eight H200s

These runs use the four-step VSA checkpoint, with all 50 transformer blocks.
The different GPU counts were measured across successive optimization builds.
The table reports observed latency rather than a controlled scaling experiment.

| H200 GPUs | 5 s request | 10 s request | Timing record |
| ---: | ---: | ---: | --- |
| 1 | 10.53–11.58 | 21.00–22.51 | [14779](tests/fasth3/evidence/14779/serve/events.jsonl) |
| 2 | 4.55–4.69 | 8.91–9.02 | [14963](tests/fasth3/evidence/14963-multigpu-2/events.jsonl) |
| 4 | 2.62–2.81 | 5.07–5.18 | [14961](tests/fasth3/evidence/14961-multigpu-4/events.jsonl) |
| 8 | 1.65–1.95 | 3.00–3.41 | [15102](tests/fasth3/evidence/15102-audio-overlap-8/events.jsonl) |

These are native-engine measurements. We have not run a matched end-to-end
comparison against the FastVideo Python application. FastH3's upstream B200
results use different hardware and settings; see the
[FastH3 release](https://haoailab.com/blogs/fasth3-preview/) for that work.

### Live: a complete one-minute streaming test

The eight-H200 browser test generated **1,433 frames, or 59.71 seconds**, across
twelve connected parts. It includes audio guidance, generation, decoding,
encoding, API delivery, and browser buffering. Model loading and service startup
are excluded from both columns below.

| Measurement | First request after startup warmup | Repeated warm prompt |
| --- | ---: | ---: |
| First playable media | 18.59 s | 2.37 s |
| Complete response received | 83.60 s | 60.04 s |
| Playback stalls after playback started | 4, totaling 11.54 s | 2, totaling 3.84 s |

Live streaming works, but playback is not yet consistently smooth. The current
adapter repeatedly decodes the accumulated latent sequence. In the warm test,
generation took about 1.54 seconds per part, while decoding grew to 5.55 seconds
for the final part. Incremental decoding is the next major improvement.

Both downloads decoded all frames without errors. These are two Chromium runs
with one repeated prompt and different seeds; they are not an iPhone hardware
test or a sustained real-time guarantee. The
[full acceptance record](tests/live/evidence/15253/acceptance.json) includes
every part's timings, playback stalls, hashes, and source revisions.
[Four-GPU short-clip timings](tests/live/evidence/15202/README.md) and
[50-second continuity checks](tests/live/evidence/15225/README.md) are also available.

## Dependencies

The core is written in C, with C++ and CUDA adapters for native libraries.
It has no Python ML framework dependency at inference time. It still needs
the GPU, tokenizer, and media libraries listed below.

| Component | Required for |
| --- | --- |
| Linux, NVIDIA driver, CUDA 12.8 toolchain | Building and running the H200 backend |
| C11 and C++17 compilers; C++20 for VSA | Native engine and GPU adapters |
| cuBLAS / cuBLASLt, cuDNN 9, NCCL | GPU math and communication |
| FlashAttention and CUTLASS headers | Compiled attention kernels |
| ThunderKittens headers | FastH3's Hopper VSA kernel |
| ICU headers and libraries | Tokenization |
| FFmpeg with H.264 and AAC support | Video/audio encoding; FFprobe is used for reference media |
| Python 3 | Dependency preparation and the optional HTTP wrappers |
| PyTorch and safetensors | One-time weight conversion on CPU; independent reference tests on GPU |

FlashAttention is compiled into the engine from pinned upstream CUDA sources;
the engine does not call its Python package. Original H3 and live generation
use FlashAttention-3 in their main attention path. FastH3 uses its trained VSA
path. Source preparation scripts verify downloaded archives by hash.

## Build and generate

Install the native dependencies above, then build all three modes:

```sh
git clone https://github.com/zeroxaa/h3.c-cuda.git H3-omni
cd H3-omni

python3 cuda/prepare_flash.py
python3 cuda/prepare_vsa.py
make -f cuda/Makefile -j8 CUDA_ARCH=sm_90 VSA=1
```

Set `CUDNN_ROOT` and `ICU_ROOT` as make variables if their locations differ from
the defaults in [cuda/Makefile](cuda/Makefile). The cuDNN default points to the
NVIDIA wheel layout in our test container; the executable links the native
library, not PyTorch. FFmpeg must be on `PATH`, or selected with `H3_FFMPEG`.

Download the original MiniMax-H3 checkpoint with its `FL2VA/` directory intact.
The [pinned download helper](tests/pytorch/download.py) uses `huggingface_hub`;
its model revision is recorded in [sources.json](tests/pytorch/sources.json).
FastH3 and live mode need separate native exports; follow
[FastH3 preparation](cuda/FASTH3.md#prepare-and-run) or
[TaoMate preparation](cuda/LIVE.md#prepare-and-run) before using those examples.
The exported directories link to base components, so keep the base weights in place.

```sh
# Original H3: complete video with audio.
build-cuda/h3 --mode original -d /models/MiniMax-H3 --gpus 1 \
  -p 'A red fox walks through fresh snow. Soft footsteps and wind.' \
  --width 864 --height 480 --seconds 5 --steps 20 --seed 7 -o original.mp4

# FastH3: the converted checkpoint selects its four-step schedule.
build-cuda/h3 --mode fast -d /models/fasth3-native --gpus 4 \
  -p 'A red fox walks through fresh snow. Soft footsteps and wind.' \
  --width 864 --height 480 --seconds 5 --seed 7 -o fast.mp4

# TaoMate-H3: two connected requests, saved as one 10.125-second clip.
build-cuda/h3 --mode live -d /models/h3-live --gpus 4 \
  -p 'A presenter talks to the camera in a bright studio.' \
  --width 864 --height 480 --frames 243 --seed 42 -o live.mp4
```

Omit `-p` to use the interactive CLI and keep weights resident between prompts.
For embedding, start with [h3.h](h3.h). The incremental session API is in
[h3_live.h](h3_live.h); it retains history between calls within one session.

## Serving and streaming

The optional service layer uses Python's standard library to handle HTTP and
supervise a resident C worker. Model inference stays inside `h3_worker`.

| Entry point | Response | Behavior |
| --- | --- | --- |
| [serve/server.py](serve/server.py), `POST /v1/videos` | Complete MP4 | Independent video requests with resident weights |
| [serve/live_server.py](serve/live_server.py), `POST /v1/live` | Growing fragmented MP4 | Twelve connected parts, one prompt, up to about one minute |

For a live worker, build the extra target and provide a bearer-key file containing
at least 32 characters. Example for a local service:

```sh
make -f cuda/Makefile -j8 CUDA_ARCH=sm_90 VSA=1 "$(pwd)/build-cuda/h3_worker"

python3 serve/live_server.py \
  --worker build-cuda/h3_worker --model /models/h3-live --gpus 8 \
  --host 127.0.0.1 --port 18961 \
  --key-file /path/to/live-key --log live-worker.log
```

`/health` reports ready after warmup. The live service admits one session at a
time, propagates client cancellation, and resets history for the next session.
One FFmpeg process produces the full stream, preserving audio/video timestamps
across parts. A browser can append incoming fragments through MediaSource or
ManagedMediaSource before generation finishes.

The Slurm launchers in `serve/` record our cluster setup and contain site-specific
paths. Adapt those paths before using them on another cluster.

## Implementation and validation

The engine combines resident weights, cached timestep conditioning, fused CUDA
operations, Ulysses sequence parallelism with NCCL, and video VAE tiles spread
across GPUs. The text encoder runs on GPU 0. Complete-clip generation overlaps
audio and video decoding; live sessions add history management and incremental
media delivery. See the [backend guide](cuda/README.md) for the execution details.

Validation is recorded separately for each mode:

- **Original H3:** the reference path matches the pinned Diffusers denoising
  trajectories bit for bit over the seven-case matrix. The faster default uses
  FA3 and an FP16 video VAE, checked against numerical acceptance thresholds.
- **FastH3:** the recorded four-step velocity and latent checks match the pinned
  same-kernel reference exactly; decoder checks use separate tolerances.
- **Live:** component checks, native determinism, historical KV, cancellation,
  and continuity have been tested. End-to-end parity with the upstream Python
  program is not established. Four-GPU results differ numerically from the
  one/two-GPU results, as documented in the evidence.

These claims apply to the recorded cases. Each model guide links to inputs,
source revisions, results, and known limitations.

| Guide | Contents |
| --- | --- |
| [CUDA backend](cuda/README.md) | Build details, C runtime, precision, multi-GPU execution, validation |
| [FastH3](cuda/FASTH3.md) | Checkpoint conversion, VSA, four-step sampling, tests |
| [Live mode](cuda/LIVE.md) | TaoMate adapter, historical KV, session API, streaming |
| [Performance work log](cuda/WORKLOG.md) | Optimization experiments, measurements, and rejected changes |
| [vLLM-Omni comparison](cuda/VLLM_OMNI_BENCH.md) | Reference stack, launch recipe, timings, and comparison limits |
| [Inherited Metal backend](METAL.md) | Original Apple Silicon implementation and CLI tutorial |

## Credits and license

H3-omni builds on [h3.c](https://github.com/antirez/h3.c) by Salvatore Sanfilippo.
We thank the teams behind MiniMax-H3, [FastVideo/FastH3](https://haoailab.com/blogs/fasth3-preview/),
and [TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3) for the models and
inference work that make these three modes possible.

The engine is distributed under the [MIT license](LICENSE). Model weights and
adapters retain their upstream licenses. Third-party code also retains its own
license, including FlashAttention, CUTLASS, ThunderKittens, the FastVideo VSA
kernel, nlohmann/json, and inherited Metal dependencies. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[the VSA notices](cuda/vendor/VSA-NOTICE.md).
