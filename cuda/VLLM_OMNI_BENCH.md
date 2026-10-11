# vLLM-Omni MiniMax-H3 on H200: warm latency, 1-8 GPUs

Measured 2026-10-11 on the DigitalOcean H200 Slurm cluster (jobs 14512 N=1,
14555 N=2, 14552 N=4, 14537 N=8). Raw data, server logs, nvidia-smi samples
and all MP4s: `/home/cj/h3-serve/vllm-omni/results/<N>gpu/`.

## Result

On one H200, vLLM-Omni is 16% faster than h3.c at 10 s (82.9 s against
98.5 s) and 20% faster at 5 s (33.0 s against 41.0 s). The whole gap is in
the video VAE. vLLM-Omni's DiT is 3-4% slower than h3.c's.
With Ulysses sequence parallelism, 8 GPUs make the 10 s request 6.1x faster
(13.6 s), and the DiT alone 6.7x faster.

## Request

864x480, `duration` 10 s (243 frames) and 5 s (124 frames), 24 fps,
`num_inference_steps=20`, `flow_shift=12`, `audio_flow_shift=3.0`, seed 7,
task `t2va`, the boat prompt. The extra parameter `"aspect_ratio": "16:9"` is
required by validation (see Failures). It does not change the canvas, because
explicit width and height take precedence. All MP4s: H.264 864x480 24 fps,
243 or 124 frames, stereo AAC.

Order per server: one 10 s warm-up, two warm 10 s requests, one first 5 s
request, two warm 5 s requests. The first 5 s request was as fast as the warm
ones (regional `torch.compile` with `dynamic=True`, so no recompile per shape).
The table uses the two warm requests.

## Warm latency (client wall time to the complete MP4)

| N GPUs | 10 s video (a / b) | 5 s video (a / b) | 10 s speedup vs N=1 | 5 s speedup vs N=1 |
| --- | --- | --- | --- | --- |
| 1 | 82.90 / 82.83 s | 33.03 / 32.96 s | 1.00x | 1.00x |
| 2 | 45.42 / 45.52 s | 18.92 / 18.96 s | 1.82x | 1.74x |
| 4 | 24.09 / 23.96 s | 10.35 / 10.21 s | 3.45x | 3.20x |
| 8 | 13.58 / 13.57 s | 6.10 / 6.06 s | 6.10x | 5.42x |
| h3.c, 1 GPU (job 14480) | 98.5 s | 41.0 s | - | - |

The two warm requests differ by at most 0.15 s.

## Stage breakdown (mean of the two warm requests, server profiler)

| N | Duration | Text encoder | DiT total | DiT per forward | Video VAE | Audio VAE | MP4 + transport | Server total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 s | 0.04 | 77.30 | 3.865 | 4.65 | 0.08 | 0.63 | 82.84 |
| 2 | 10 s | 0.04 | 41.85 | 2.092 | 2.53 | 0.07 | 0.61 | 45.46 |
| 4 | 10 s | 0.04 | 21.61 | 1.081 | 1.33 | 0.07 | 0.58 | 24.02 |
| 8 | 10 s | 0.04 | 11.61 | 0.580 | 0.77 | 0.07 | 0.59 | 13.57 |
| 1 | 5 s | 0.04 | 30.15 | 1.507 | 2.32 | 0.04 | 0.33 | 32.95 |
| 2 | 5 s | 0.04 | 17.05 | 0.852 | 1.26 | 0.04 | 0.33 | 18.93 |
| 4 | 5 s | 0.04 | 8.97 | 0.448 | 0.68 | 0.04 | 0.30 | 10.27 |
| 8 | 5 s | 0.04 | 4.98 | 0.249 | 0.44 | 0.04 | 0.32 | 6.07 |

All times in seconds. "DiT total" is the profiler's `diffuse` stage (the
whole denoise loop). "MP4 + transport" is server total minus the stage-0
generation time. The profiler adds one device synchronize around each
profiled stage; that cost is small against these times.

Step count: vLLM-Omni runs 20 DiT forwards for `num_inference_steps=20`
(progress bar 20/20; the code builds N+1 sigma boundaries for N forwards).
The blog's "50 sigma points = 49 forwards" is its own 50-point request, not a
rule.

DiT scaling (per-forward time against N=1):

| N | 10 s video | 5 s video |
| --- | --- | --- |
| 2 | 1.85x (92% efficiency) | 1.77x (88%) |
| 4 | 3.58x (89%) | 3.36x (84%) |
| 8 | 6.66x (83%) | 6.05x (76%) |

Shorter sequences scale less, because the all-to-all and the per-rank fixed
work take a larger share.

## Comparison with h3.c on one GPU

| Stage | h3.c 10 s | vLLM-Omni 10 s | h3.c 5 s | vLLM-Omni 5 s |
| --- | --- | --- | --- | --- |
| Denoise (20 forwards) | 75.0 s | 77.3 s | 29.0 s | 30.1 s |
| Audio VAE | 0.25 s | 0.08 s | 0.18 s | 0.04 s |
| Video VAE + MP4 | 23.1 s | 5.3 s | 11.7 s | 2.7 s |
| Request | 98.5 s | 82.9 s | 41.0 s | 33.0 s |

h3.c numbers are from `/home/cj/h3-serve/duration-14480.log` (phase
timestamps). The main difference is precision: vLLM-Omni decodes the video
VAE under FP16 CUDA autocast over FP32 weights; h3.c uses 3xTF32 (close to
FP32). I did not compare the two outputs pixel by pixel.

## Peak GPU memory (nvidia-smi, 0.5 s sampling)

| N | Rank 0 | Other ranks | PyTorch peak (response header) |
| --- | --- | --- | --- |
| 1 | 127.4 GB | - | 126.7 GB |
| 2 | 125.9 GB | 76.5 GB | 123.1 GB |
| 4 | 124.6 GB | 75.4 GB | 121.0 GB |
| 8 | 125.7 GB | 76.5 GB | 120.6 GB |

Model weights on one GPU: 115.6 GiB (DiT + Qwen3-VL text encoder + VAEs);
everything fits on one H200 without offload. Rank 0 holds the text encoder
(the default `--text-encoder-tp-size` is 1), so it uses about 50 GB more than
the other ranks. Every rank holds the full DiT (pure Ulysses). All GPUs ran
at the 700 W power cap (peaks 699-717 W).

Startup from node-local disk: 58 s (N=1), 87 s (N=2), 119 s (N=4), 193 s
(N=8) until `/health`. First request (includes compile): 95.6 s, 55.4 s,
39.8 s, 58.0 s for the 10 s video.

## Launch flags (exact)

```
vllm serve /var/tmp/h3omni/root --omni --trust-remote-code --num-gpus N --usp N --ring 1 \
  --vae-patch-parallel-size N --vae-parallel-mode tile --vae-use-tiling --task-type fl2va \
  --enable-diffusion-pipeline-profiler --init-timeout 3600 --port 810N
```

N in {1, 2, 4, 8}. No CPU offload, no `--enforce-eager`, default attention
backend (log: `Resolved diffusion attention backend 'FLASH_ATTN'`),
`--cfg-parallel-size` left at 1.

Environment: `LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH`,
`VLLM_ENABLE_CUDA_COMPATIBILITY=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
`VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200`, `HF_HUB_OFFLINE=1`, static ffmpeg and
ffprobe on `PATH`.

## Install and launch recipe that worked on this cluster

Scripts: `/home/cj/h3-serve/vllm-omni/{install.sh,bench.sh,bench.sbatch,client.py,summarize.py}`.

1. Container: `/data0/apptainer/vllm_glm52.sif` (Python 3.12, `uv`, CUDA 13
   forward-compat `libcuda` 580.95 in `/usr/local/cuda/compat`).
2. Venv on node-local disk (`install.sh`, 80-90 s):
   ```
   uv venv -p /usr/bin/python3 /var/tmp/h3omni/venv
   uv pip install vllm==0.31.0 --torch-backend=cu129 \
     --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
   # vllm-omni main 194b936637f4e7356cc998c2796c67d14d52d854 (2026-10-10)
   VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=<version> uv pip install -e .
   ```
   Result: vLLM 0.31.0, torch 2.13.0+cu129, transformers 5.14.1,
   diffusers 0.40.0.
3. Weights: copy `FL2VA/` and the root `model_index.json` from
   `/data0/models/h3-cuda/42ed227e...` to `/var/tmp/h3omni/root/` with
   `xargs -P 16 cp -L` (135 GB, 43-85 s). Do not copy
   `modular_model_index.json`.
4. Submit with arguments, not `--export`:
   `sbatch -p h200 -w <node> --gres=gpu:N -c 24 --mem=400G bench.sbatch N`
   (N=8: `-p train --gres=gpu:8`).
5. Request (multipart `POST /v1/videos/sync`): `prompt`, `width=864`,
   `height=480`, `fps=24`, `num_inference_steps=20`, `flow_shift=12`,
   `seed=7`, `extra_params={"task":"t2va","duration":10,"audio_flow_shift":3.0,"aspect_ratio":"16:9"}`.
   With `--enable-diffusion-pipeline-profiler` the response carries
   `X-Stage-Durations`, `X-Inference-Time-S` and `X-Peak-Memory-MB` headers.

## What failed and how I fixed it

| Problem | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: vllm.entrypoints.serve.exception_handling` | The recipe pins `vllm==0.26.0`, but vLLM-Omni `main` needs vLLM 0.31.0 (its own docs) | vLLM 0.31.0 from the commit-pinned wheel index, `--torch-backend=cu129` |
| Editable install fails: `Invalid version: 'dev'` | Shallow clone has no git tags for setuptools_scm | `VLLM_OMNI_VERSION_OVERRIDE`. I set `0.26.0.dev0+g194b936`, which only causes a harmless "mismatched versions" warning; use a 0.31-based string next time |
| `No such file: .../fastvideo_inference.json` | A root with `modular_model_index.json` is treated as a FastH3 (FastVideo) release. The base model is read from `<root>/FL2VA/` | Serve a root with `FL2VA/` and `model_index.json` only. The top-level `transformer/`, `vae/`, `audio_vae/`, schedulers I downloaded (sha256 verified, in the shared snapshot) are not used by vLLM-Omni |
| `t2va requires an explicit aspect_ratio` | Validation needs a named ratio even when width and height are given | `"aspect_ratio": "16:9"` in `extra_params`; canvas stays 864x480 |
| `cudaGetDeviceProperties failed: CUDA driver version is insufficient` in `flash_attn` | `fa3-fwd` kernels need a CUDA 13 driver; node driver is CUDA 12.9 | Container forward-compat libs: `LD_LIBRARY_PATH=/usr/local/cuda/compat` |
| Jobs held: `user env retrieval failed requeued held` | `sbatch --export=ALL,VAR=...` | Pass values as script arguments |
| `vllm._C` import failure during install | Install job had no GPU (no `--nv`) | None needed; works in GPU jobs |

CPU offload was never needed.

## vLLM-Omni optimizations worth porting to h3.c

1. **FP16-autocast video VAE decode (largest single-GPU win).** diffusers
   0.40.0 `autoencoder_kl_minimax_h3.py` says "the verified decode recipe is
   float16 *autocast over float32 weights*", and
   `modular_pipelines/minimax_h3/decoders.py:187` wraps decode in
   `torch.autocast(dtype=torch.float16)`. vLLM-Omni follows that
   (`pipeline_minimax_h3.py` decode, `ops/vae` keeps decoder Linear weights in
   FP16). Evidence: video VAE 4.65 s against about 23 s in h3.c at 10 s.
   h3.c's reference test (`tests/pytorch/decoders.py`) decodes in plain FP32,
   so h3.c treats a stricter target than the official recipe. Moving to FP16
   tensor-core GEMMs and attention in the VAE, validated against a PyTorch
   FP16-autocast reference, could save about 18 s at 10 s (estimate, not
   measured).
2. **Ulysses sequence parallelism for the DiT.** Measured 6.66x DiT speedup
   on 8 H200 at 10 s, 3.58x on 4. Each rank holds the full DiT, shards the
   token sequence, and uses an all-to-all to trade sequence shards for head
   shards around attention (56 heads, so N must divide 56).
3. **VAE tile (patch) parallelism across ranks.** Video VAE 4.65 s on 1 GPU,
   0.77 s on 8.
4. Already in h3.c, so no gain expected: fused QK-norm+RoPE, fused SwiGLU,
   fused modulation and residual (h3.c #9), exact AdaLN reuse (h3.c
   precomputes AdaLN), resident text encoder (both 0.03-0.04 s), streamed MP4.
   On one GPU h3.c's DiT is already 3% faster per forward (3.75 s against
   3.87 s at 10 s).
5. Lossy options in vLLM-Omni, not measured here: online FP8 DiT (blog, B300:
   5.3% faster, 39% less HBM), SAGE FP8 attention and Skip-Softmax
   (`TRTLLM_ATTN`, Blackwell only), Cache-DiT (blog: about 30% less denoise,
   SSIM 0.831), and the FastH3 four-step LoRA.

## Limits of this measurement

- One prompt and one seed; two warm requests per point.
- The profiler synchronizes around stages; the client wall times include it.
- I did not compare output quality between vLLM-Omni and h3.c, or between N
  values. MP4 sizes differ slightly between N, so the outputs are not
  bit-identical across parallel layouts.
- N=1 ran on gpu-h200x8-2; N=2, 4, 8 on gpu-h200x8-3. h3.c's 98.5 s is from
  an earlier job (14480), not a same-node A/B.
