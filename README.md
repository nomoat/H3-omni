# h3.cu

**A specialized inference engine for MiniMax-H3, in C and CUDA, for H200.**

h3.cu runs one model family: MiniMax-H3 text-to-video+audio and the four-step FastH3 Preview v1 distillation. It is
written for one target: Hopper GPUs (H200, `sm_90`), one node, 1 to 8 GPUs,
in a long-lived serving process.

It is not a general framework and not a local, portable tool. There is no
Python, PyTorch or libtorch at run time. Every kernel, buffer and schedule is
chosen for H3's shapes. Python is used only to build reference results and run
differential tests.

## Performance

Warm requests in one resident process, 864x480, 24 fps, with audio, MP4
encoding included. H200 nodes of the documented Slurm cluster.

**MiniMax-H3, 20 steps:**

| Video | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
| --- | --- | --- | --- | --- |
| 5 s | 32.0 s | 18.1-18.3 s | 9.2-9.4 s | 5.4-5.7 s |
| 10 s | 80.4 s | 44.2-44.7 s | 22.2-22.4 s | 12.6-13.1 s |
| vLLM-Omni, 5 s / 10 s | 33.0 / 82.9 s | 18.9 / 45.4 s | 10.2 / 24.0 s | 6.1 / 13.6 s |

Jobs 14771, 14781, 14789, 14792. These predate the GPU-side VAE output (#17
in the work log), which mainly helps at 8 GPUs.

**FastH3, 4 steps:**

| Video | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
| --- | --- | --- | --- | --- |
| 5 s | 10.5-11.6 s | 4.55-4.69 s | 2.62-2.81 s | **1.65 s** |
| 10 s | 21.0-22.5 s | 8.91-9.02 s | 5.07-5.18 s | **3.00-3.03 s** |

Jobs 14779, 14963, 14961, 15102. At 8 GPUs a 5 s video is generated in a
third of its play time: 0.11 s text and preparation, 0.83 s denoising, about
0.5 s video VAE (audio VAE in parallel), 0.2 s MP4 tail.

These are individual measured requests, not a statistical benchmark. The GPUs
run at their 700 W power cap, and times vary a little between nodes.

## Correctness

- **Reference mode** (`H3_FLASH3=0 H3_VAE_3XTF32=0`) matches the pinned
  PyTorch/Diffusers pipeline bit for bit over a 7-case, 20-step matrix.
- **Default mode** uses FlashAttention-3 in the DiT and FP16 tensor cores in
  the video VAE. It passes a calibrated acceptance; decoded video error is
  4e-5 to 3e-4 relative L2 against the FP32 reference.
- **Multi-GPU output is bit-identical to one GPU** (latents, audio, RGB and
  MP4 bytes) for N = 2, 4, 7 and 8, for H3 and FastH3.
- Every speed change is checked bit for bit against the previous build, or
  against the calibrated limits when it changes arithmetic.

Details: [cuda/README.md](cuda/README.md). Every experiment, including the
ones that failed: [cuda/WORKLOG.md](cuda/WORKLOG.md).

## How a request runs

```text
prompt ─► tokenizer (CPU) ─► Qwen text encoder (GPU 0)
                                  │
                                  ▼
          DiT, 50 blocks × steps, Ulysses sequence parallel on N GPUs
          (NCCL all-to-all around attention, Euler sampler on GPU)
                                  │
                ┌─────────────────┴──────────────────┐
                ▼                                    ▼
   video VAE: tiles on GPU i % N,          audio VAE (last GPU,
   NVLink to GPU 0, stitch/blend/RGB       in parallel)
   on GPU 0                                          │
                │                                    │
                └──────────► x264 + AAC (CPU, FFmpeg) ◄┘ ─► MP4
```

The only real CPU work is the MP4 encode: the H200 has no NVENC. It runs
while the VAE decodes, so only about 0.2 s remains after the last frame.

Main techniques:

- All weights resident; a new prompt, seed or canvas reuses them.
- DiT: FlashAttention-3, fused bit-exact elementwise kernels, a cached AdaLN
  table, Ulysses sequence parallelism over 1, 2, 4, 7 or 8 GPUs.
- FastH3: the Hopper VSA (sparse video attention) kernel, split by heads.
- Video VAE: fixed 256-pixel tiles in a 4-slot pipeline, FP16 tensor cores
  with FP32 norms and accumulation, all post-processing on the GPU.

## Build and run

Requirements: CUDA 12.8, cuDNN 9, NCCL, a C/C++17 toolchain (C++20 for the
VSA kernel), ICU, and FFmpeg at run time. The documented Apptainer image
`pytorch_291_cu128.sif` has all but ICU and FFmpeg.

```sh
python3 cuda/prepare_flash.py      # pinned FlashAttention/CUTLASS sources
python3 cuda/prepare_vsa.py        # only for FastH3 (VSA=1)
make cuda CUDA_ARCH=sm_90          # MiniMax-H3
make -f cuda/Makefile -j16 VSA=1   # MiniMax-H3 and FastH3
```

```sh
build-cuda/h3 -d /path/to/MiniMax-H3 --gpus 8 \
  -p 'A red fox walks through fresh snow in a pine forest.' \
  --width 864 --height 480 --frames 120 --steps 20 --seed 7 -o fox.mp4
```

Without `-p` the binary starts an interactive session that keeps every model
resident between prompts. FastH3 needs a converted checkpoint; see
[cuda/FASTH3.md](cuda/FASTH3.md).

## HTTP service

`serve/` turns the engine into a small video API for one node:

- `h3_worker` (C) loads the model once, keeps it resident on all GPUs and reads
  one request per stdin line.
- `server.py` (Python standard library only) puts it behind
  `POST /v1/videos` → `video/mp4`, with a bearer key, a short queue (429 when
  full) and a warm-up generation before `/health` reports ready.
- `h3-video.sbatch` builds and runs both as a Slurm job with no time limit.

```sh
sbatch --gres=gpu:8 serve/h3-video.sbatch src.tar 8 18960
curl -H "Authorization: Bearer $KEY" -d '{"prompt":"A fox in snow.","seconds":5}' \
  http://NODE:18960/v1/videos -o fox.mp4
```

It backs the video model in the autograd.run playground. Measured through
that path on 8 H200 (FastH3, 864x480, warm): 2 s video 1.15 s, 5 s video
1.8 s, 10 s video 3.5 s of GPU time per request. The first request after
start takes about 2 minutes to load the weights; the warm-up absorbs it.

## Documents

| File | Content |
| --- | --- |
| [cuda/README.md](cuda/README.md) | Correctness contract, serving residency, multi-GPU, tests |
| [cuda/FASTH3.md](cuda/FASTH3.md) | FastH3 checkpoint conversion, VSA kernel, results |
| [cuda/WORKLOG.md](cuda/WORKLOG.md) | Every speed experiment: change, measurement, verdict, Slurm job |
| [METAL.md](METAL.md) | Upstream Apple Silicon backend and its CLI tutorial |

## Origin and license

h3.cu started from [antirez/h3.c](https://github.com/antirez/h3.c), a native
MiniMax-H3 implementation for Apple Silicon. The model code in C is shared;
the CUDA backend, multi-GPU support, serving path and FastH3 support are added
here. The Metal code is still in the tree but is not maintained by this fork.

MIT license, see [LICENSE](LICENSE). Third-party code keeps its own license:
the FastVideo VSA kernel and PyTorch reduction order (`cuda/vendor/`),
FlashAttention, CUTLASS and ThunderKittens (fetched into `.deps/` by the
prepare scripts), nlohmann/json, and ccv for the Metal code
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

Not affiliated with MiniMax or FastVideo.
