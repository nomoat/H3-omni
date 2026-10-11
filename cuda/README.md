# H200 correctness backend

This backend keeps h3.c's C model code and uses CUDA, cuBLAS, cuDNN and the
FlashAttention forward kernel bundled with the pinned PyTorch reference. The
executable has no Python, PyTorch or libtorch runtime dependency. Python is used
only to produce independent reference results and run differential tests.

**Phase one passed the defined text-to-video/audio matrix and the default-size
held-out case on H200**, Slurm job 14274 (2026-10-10). This is a correctness
baseline; performance optimization is a separate stage. The standalone executable
also produced a valid 22-frame MP4 with audio. Its runtime dependencies contain
neither libtorch nor libpython.

**Two modes.** The default mode is for serving: FlashAttention-3 for the DiT
attention, and FP16 tensor-core products and FP16 FlashAttention-3 in the
video VAE, with FP32 norms, residual stream and accumulation
(`vae_fp16.cuh`; decoded error 4e-5 to 3e-4 against the PyTorch FP32 decode,
6x to 10x less than the diffusers FP16 decode). `H3_VAE_PRECISION=3xtf32`
selects the previous 3xTF32 video VAE (`vae_attention.cuh`, error about
1e-4). The reference mode,
`H3_FLASH3=0 H3_VAE_3XTF32=0`, is the phase-one backend that matches PyTorch
bit for bit; it must keep passing the contract below. The default mode passes
the calibrated acceptance described in [WORKLOG.md](WORKLOG.md), which also
records every latency experiment, what worked and what did not.

| Check | Result |
| --- | --- |
| CUDA primitives | 69 passed |
| Tokenizer | 10 cases, exact token IDs |
| Text conditioning | English, Chinese and 82-token prompt: exact |
| Sigma grids / sampler / layout | 12 exact sampler cases; 32 exact layout cases |
| Seven independent, 20-step trajectories | All 560 comparisons exact |
| Decoded video, worst relative L2 | `1.287374e-06` |
| Decoded audio, worst relative L2 | `4.440359e-05` |
| Standalone CLI | 256x256, 22 frames, audio present |

The required matrix has six short-video cases: 256x256 and 512x512, 22 frames,
seeds 0, 1 and 42. The additional case uses an 82-token prompt, 864x480 pixels,
56 frames and seed 7. All cases use the original weights, 50 DiT layers and
20 model evaluations. The decoded normalized maximum errors are
`2.232194e-05` (video) and `8.194175e-05` (audio).
All 28 decoded comparisons passed the unchanged relative L2 limit of `0.05`
and normalized maximum limit of `0.1`. These results cover the stated matrix,
not every prompt, geometry or library version.

Compact reports are in [`tests/pytorch/evidence/14274`](../tests/pytorch/evidence/14274),
including `summary.json`, `acceptance-result.json`, `extended/results.json` and
`source-verification.json`. Complete fixtures, per-step tensors, decoded tensors,
build source and logs are retained at `/home/cj/h3-cuda/results/14274`.
The compact local copy omits large tensors; run `accept.py` against the complete
cluster directory. The current 96 implementation/build/test files match the
archived build source. Documentation and compact reports are added afterward.

## Contract

This contract is for the reference mode. The default mode deviates only in
the two kernels named above.

- Original FL2VA weights; text-to-video/audio; 50 DiT blocks; all requested
  model evaluations. Image/keyframe conditioning and Ref2VA are outside the
  current acceptance matrix.
- No layer skipping, step reuse, core reuse, token merging or quantization.
  Match the reference's fixed 256-pixel VAE tiles and temporal chunking.
- FP32 accumulation without TF32. Preserve the checkpoint's mixed precision.
- Match PyTorch's BF16 operation boundaries, including SiLU before the timestep
  embedding cast. Keep FP32 final heads and FP32 solver state.
- Same serialized input noise, token IDs, layout and sigma schedule.
- Compare tensor values before video/audio encoding. MP4 byte equality is not
  a model-correctness criterion.
- Thresholds are in `tests/pytorch/acceptance.json`, established before model
  measurements. Missing fixtures, non-finite values and missing stages fail.

The public reference is the implementation linked by MiniMax's model card:
[Diffusers](https://github.com/huggingface/diffusers). Its revision and the
original model snapshot are pinned in `tests/pytorch/sources.json`. Reference
code is read from that checkout, not rewritten to mirror the CUDA backend.
The upstream MLX fixture generators are absent from this h3.c checkout.

## Build

Requirements: CUDA 12.8, cuDNN 9, a C/C++17 toolchain and ICU headers/libraries.
Use a GPU allocation and the container specified in `../GPU-ACCESS.md` (workspace
file, outside this repository). Stage source and build output on node-local disk.

```sh
python3 cuda/prepare_flash.py  # Prepare pinned headers before allocating a GPU.
make cuda CUDA_ARCH=sm_90
```

`cuda/Makefile` accepts `CUDNN_ROOT` and `ICU_ROOT`. Its default cuDNN path matches
the documented PyTorch container. `cuda/prepare_icu.sh` downloads ICU packages
without modifying the container. Run dependency preparation on the login node.

Build outputs are `build-cuda/h3`, `libh3_cuda.so` (primitives), `libh3_model.so`
(test interface) and `libh3_tokenizer.so`. If copying a binary to another node,
also make its ICU runtime available through `LD_LIBRARY_PATH`.

Run the standalone executable inside the CUDA container:

```sh
build-cuda/h3 -d /path/to/checkpoint -p 'A cat walks through a sunny garden.' \
  --width 256 --height 256 --frames 22 --steps 20 --layers 50 \
  --reuse 1 --core-reuse 1 --seed 0 -o cat.mp4
```

The model directory contains `FL2VA/`. FFmpeg must be on `PATH`, or its binary
path must be supplied through `H3_FFMPEG`. The executable generates its own
initial noise. Differential tests instead load an identical serialized noise
fixture into both implementations.

The accepted binary, shared libraries, ICU runtime, FFmpeg and a container
launcher are retained at `/home/cj/h3-cuda/releases/14274`. Inside a single-H200
Slurm allocation, the launcher can run the same example:

```sh
/home/cj/h3-cuda/releases/14274/run.sh \
  -d /data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe \
  -p 'A cat walks through a sunny garden.' \
  --width 256 --height 256 --frames 22 --steps 20 --layers 50 \
  --reuse 1 --core-reuse 1 --seed 0 -o cat.mp4
```

## Serving residency

A one-shot `h3 -p ...` process loads every weight, generates once and exits.
That is the path the acceptance below measures. For serving, a long-lived
process should load weights once. With the cache enabled (interactive CLI, or
`h3_cache_set_enabled` in the library), CUDA builds keep these objects loaded
across requests by default (`--resident-weights off` or
`h3_cache_set_resident_weights(ctx, 0)` disables it):

| Object | Resident content | Rebuilt per request |
| --- | --- | --- |
| Tokenizer | parsed `tokenizer.json` | token IDs |
| Qwen text encoder | 50 layers and the token embedding | prompt activations |
| DiT | 50 blocks, patch/final heads, token refiner, last AdaLN table | refined text, RoPE, row maps, activations; AdaLN table only when the step count or condition kinds change |
| Video VAE decoder | all weights | tile plan, RoPE and tile activations when the canvas changes |
| Audio VAE decoder | weight-normalized convolutions | activations |

A new prompt, seed, canvas, frame count or step count runs the same request
preparation code as a fresh load (`h3_dit_rebind` reuses it), so results do not
depend on what the process served before. The AdaLN table depends only on the
sigma schedule, the condition kinds and the active blocks. Its projections are
26 GB, so the table (0.4 GB at 20 steps) is cached instead of the weights; a
new step count re-reads them once. Peak use on one H200 is about 100 GB.
Image/video/audio-conditioned requests still stream their Qwen vision and
encoder weights; only their DiT and decoders are resident.

`tests/serving/serve.sbatch` checks this. One process serves seven requests:
two prompts at 864x480/56 frames, the first prompt again after the second,
512x512/22 frames, the second prompt again after the canvas change, a Chinese
prompt at 256x256 with 12 steps, and the second prompt again at 20 steps. Each
distinct request is also run as a fresh one-shot process. The denoised video/audio latents, decoded RGB and PCM
(`H3_DUMP_DIR`, diagnostic only) and MP4 bytes must be identical. The same job
reruns the full acceptance above on the new build and compares the one-shot
MP4 with release 14274.

Slurm job 14416 (2026-10-10, one H200, this source) passed every check:
all 7 serving requests were bit-identical to their fresh one-shot references;
the one-shot MP4 equals release 14274's MP4; and the PyTorch acceptance, the
held-out 864x480/56-frame case and the CLI smoke test passed unchanged. GPU
memory peaked at 99,169 MiB. Compact reports are in
[`tests/serving/evidence/14416`](../tests/serving/evidence/14416).

| Request | Fresh process (s) | Resident process (s) |
| --- | --- | --- |
| 864x480, 56 frames, 20 steps, first request | 88.5 | 94.0 (loads every weight) |
| 864x480, 56 frames, 20 steps, new prompt | 89.4 | 33.0 |
| 512x512, 22 frames, 20 steps | 63.6 | 9.6 |
| 256x256, 22 frames, 12 steps (new AdaLN table) | 55.8 | 10.6 |
| 864x480, 56 frames, back to 20 steps (new AdaLN table) | — | 40.3 |

A warm 864x480 request is now 17.4 s of denoising and about 15 s of video VAE
decoding; text, refinement and audio take less than 0.3 s together. Times
include MP4 encoding. Fresh-process times include process start, weight
loading and the same work.

Longer videos, Slurm job 14419 (`tests/serving/duration.sbatch`, one
resident process, 864x480, 20 steps, three prompts each, warm):

| Requested | Output | Mean (s) | Denoise (s) | Audio+video VAE+MP4 (s) |
| --- | --- | --- | --- | --- |
| 5 s (120 frames) | 124 frames, 5.17 s | 81.7 | 46.1 | 35.4 |
| 10 s (240 frames) | 243 frames, 10.13 s | 201.3 | 130.7 | 70.4 |

Peak GPU memory was 104 GB. Frame counts align up to 5 + 17k. Before this
run, audio longer than about 8.2 s failed with "invalid configuration
argument": `h3_gpu_dispatch_3d` put the sample axis in grid y, which CUDA
limits to 65535 blocks. Taller grids now use taller blocks. Grids inside the
limit keep the old shape, and the 5 s MP4s are byte-identical to job 14418
from before the change.

## Serving latency

| Request (warm, 864x480, 20 steps) | Reference mode, job 14419 | Default mode, job 14771 | vLLM-Omni, 1 GPU |
| --- | --- | --- | --- |
| 5 s video | 81.7 s | 32.0 s | 33.0 s |
| 10 s video | 201.3 s | 80.4 s | 82.9 s |

Job 14419 predates the stream-ordered memory pool, which is bit-exact and on
in both modes. The FP16 video VAE keeps 4.8 GB of block weights; the 3xTF32
one keeps 19 GB (a TF32 high and low part of its four block projections). Details, profiles and
rejected experiments: [WORKLOG.md](WORKLOG.md).

The resident video VAE decodes its tiles as a pipeline: the GPU works on the
next tile while a host thread rearranges, blends and normalizes the previous
one, and FFmpeg encodes each finished chunk during the decode. Both modes
give the same bytes as the serial path (job 14473/14474 bit checks); a warm
10 s request spends 32.1 of its 33.0 s video VAE on the GPU (job 14476).

## Deployment requirements

- Weights: `FL2VA/` (prompt-only and first/last-frame requests). Ordered
  references (`--ref-image`, `--ref-video`, `--ref-audio`, ...) also need the
  complete `Ref2VA/` directory, 144 GB: h3 reads its tokenizer, text encoder,
  transformer and both VAEs from there, not only the transformer.
  `python3 tests/pytorch/download.py --ref2va DIR` fetches both from the
  pinned revision. At revision 42ed227e only `Ref2VA/transformer` differs
  from FL2VA; the other files are byte-identical.
- FFmpeg and FFprobe: `H3_FFMPEG` encodes the MP4; `H3_FFPROBE` reads the size
  and streams of reference images and videos (default: `ffmpeg`/`ffprobe`
  on `PATH`). The cluster uses the johnvansickle.com 7.0.2 static build for
  both.
- Peak GPU memory in resident serving: 118 GB at 1344x768, 768x1344 and
  15 s (job 14488).

## Reference environment and tests

`tests/pytorch/prepare.sh` installs the pinned Diffusers source and Transformers
in an isolated environment, disables user-site packages and records the resolved
dependency versions. It runs on the login node without a GPU. Reference source
and dependencies are archived for node-local extraction.

```sh
python3 tests/pytorch/primitives.py --output primitives.json
python3 tests/pytorch/host.py --output host.json
python3 tests/pytorch/text.py --weights /path/to/checkpoint/FL2VA --output text
python3 tests/pytorch/real_dit.py \
  --weights /path/to/checkpoint/FL2VA --diffusers /path/to/pinned/diffusers \
  --text text/text-0.safetensors --width 256 --height 256 --frames 22 \
  --steps 20 --seed 0 --output dit
python3 tests/pytorch/decoders.py \
  --weights /path/to/checkpoint/FL2VA --diffusers /path/to/pinned/diffusers \
  --latents dit/latents.safetensors --output decoded
```

`--steps 20` means 20 model evaluations. The Diffusers scheduler receives 21
grid points because it includes the terminal zero. `--evaluations` runs only a
diagnostic prefix; such a run cannot count as trajectory acceptance. Synthetic
text embeddings are also diagnostic only.

The reference calls the public pipeline's `build_row_timesteps` helper at every
step. It uses one distinct timestep row at the first evaluation and two at the
following evaluations. The CUDA precompute preserves each evaluation's matrix
shape; merging all evaluations into one GEMM can change floating-point rounding.
The acceptance gate requires this official timestep plan in every trajectory
report. Historical job 14269 used duplicate first-step rows and is superseded;
job 14271 records the resulting regression against the corrected reference.
Job 14273 then passed the six-case matrix but exposed a non-square FP64 spatial
position difference in the held-out case. The CUDA layout now uses NumPy's
rounded-endpoint linspace operation order; both rectangular orientations are
covered by exact regression checks. Job 14274 includes both fixes.

The required matrix is two canvases (256x256 and 512x512), 22 frames, three
seeds (0, 1 and 42), 50 layers and 20 model evaluations. These are short module
tests; they do not claim validation of every shape or the full public pipeline's
duration range. Input fixtures are serialized once and used by both paths;
matching a seed alone does not imply matching random tensors across libraries.

`trajectory_suite.py` and `decoder_suite.py` run this matrix.
`cuda/acceptance.sbatch` builds the executable, runs the matrix, verifies a real
MP4 from the standalone CLI, and invokes `accept.py`. The final check rejects
missing stages, diagnostic runs and absent artifacts. It does not treat a
successful build or a passing single forward as acceptance.

The same job also runs `extended.py`, using the held-out configuration in
`tests/pytorch/extended.json`: an 82-token prompt, 864x480 pixels, 56 frames,
seed 7 and 20 evaluations. This checks a longer text sequence, a rectangular
canvas and three overlapping temporal VAE chunks under the original limits.
It produces a separate `extended/results.json` and full tensor artifacts.

`cuda/build.sbatch` runs compilation and primitive checks. `cuda/parity.sbatch`
currently runs a small real-weight DiT diagnostic. Both request one GPU. If the
`compute` partition is busy, the mixed `h200` partition can supply one GPU; do
not request an eight-GPU training allocation for these tests.

The current implementation deliberately favors inspectable operation boundaries.
BF16 attention with head dimension 128 uses the same upstream FlashAttention
forward source as PyTorch 2.9.1. Other shapes use bounded query chunks with
cuBLAS and softmax kernels. This choice aligns numerical behavior with the
reference. It does not fuse attention with adjacent model operations.
Performance measurements must separate model loading, forward computation,
denoising and decoding. Fusion and other performance work come after acceptance.
Text/decoder reports label their combined load-and-compute time explicitly.
DiT reports separate loading from synchronized forward time. Instrumented
`--trace` runs are diagnostic and are excluded from performance acceptance.

For numerical investigation, `real_block.py` replays individual operations on
actual checkpoint weights and captured activations. The default PyTorch SDPA
path differs numerically from its FP32 math path. An explicit
`--attention-math-diagnostic` run can measure this difference, but cannot count
as acceptance against the default reference.

`real_heads.py` replays the final normalization and output projections from a
captured hidden state. Qwen normalization, DiT normalization, timestep batch
shape and the FP32 projection's bias epilogue preserve the reference's operation
order. These details can affect the independent sampling trajectory even when
a single forward difference is small. The normalization reduction order follows
[PyTorch 2.9.1](https://github.com/pytorch/pytorch/tree/v2.9.1/aten/src/ATen/native/cuda);
its license is retained in `cuda/vendor/PYTORCH-LICENSE`.

The VAE uses the reference's fixed 256-pixel tiles. Pixel normalization and
clipping happen after spatial and temporal blending, which preserves overlap
values. The Metal backend's adaptive tile sizes and earlier clipping remain
unchanged.

The accepted harness recorded median native forward times of
0.207 s (256x256) and 0.301 s (512x512),
versus 0.058 s and 0.169 s for PyTorch.
Loading is excluded. Native timings include host input/output transfers; PyTorch
returns CUDA tensors. These are harness measurements, not a matched throughput
benchmark or evidence of a speedup. The standalone CLI took
58.6 s including loading, text encoding,
20 evaluations, decoding and media output with the tested cache state.

## Source maintenance

`cuda/translate_primitives.py` generates the scalar CUDA kernels and wrappers
from the pinned Metal code. Explicit BF16 rounding adjustments are recorded in
the generator. Regenerate with `python3 cuda/translate_primitives.py`.

`cuda/fast_primitives.cuh` is hand-written and is not touched by the
generator. It holds bit-identical fast versions of the DiT elementwise kernels
(AdaLN, gate, gate fused into the next AdaLN, SwiGLU, q/k RMS norm + RoPE):
vector loads and stores, one warp per attention head, and the same per-element
operations, rounding points and reduction order as the generated kernels. The
host wrappers in `h3_gpu_cuda.cu` select them when shape and alignment fit and
otherwise call the generated kernels; `H3_GENERATED_ELEMENTWISE=1` forces the
generated kernels. The generator skips the `h3_gpu_gate_bf16` and
`h3_gpu_swiglu_bf16` wrappers for this reason. If the generator changes the
arithmetic of one of these kernels, change the matching fast kernel too, then
run `tests/serving/profile/elementwise.sbatch`, which compares every fast
kernel bit for bit with the generated one.

Optional Metal/int8/token-reduction entry points return explicit errors. They
are not fallback implementations and are outside this phase's numerical scope.

The portable tokenizer uses ICU and the vendored nlohmann/json header 3.11.3
(`cuda/vendor/json.hpp`, MIT license retained in the file).

`cuda/prepare_flash.py` verifies pinned source archive digests and preserves
FlashAttention and CUTLASS licenses under `.deps/flash-cuda` and
`.deps/cutlass-flash`. Only unused ATen RNG plumbing is removed; dropout is
disabled. No ATen or libtorch code is linked. The adapter instantiates only
the BF16 head-dimension-128 forward kernels needed by H3. Cluster scripts
expect these two directories archived as `flash-deps.tar.gz` in the staging
root. Revisions and archive digests are recorded in `tests/pytorch/sources.json`.
