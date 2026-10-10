# Serving latency work log

Goal: lower the latency of one warm serving request on one H200. "Lossless"
first: no fewer steps, layers or tokens, and no approximation of the model.

All times are warm requests in one resident process at 864x480, 20 steps,
unless stated otherwise. Evidence paths are relative to `tests/serving/evidence/`
(compact copies) and `/home/cj/h3-serve/results/<job>` on the cluster (full).

## Current result

| Request | Start (job 14419) | Now (job 14436) | Change |
| --- | --- | --- | --- |
| 2.3 s video (56 frames) | 33.0 s (job 14416) | 20.8 s (job 14438) | -37% |
| 5 s video (124 frames) | 81.7 s | 51.0 s | -38% |
| 10 s video (243 frames) | 201.3 s | 118.7 s | -41% |

Two modes:

| Mode | Settings | Matches PyTorch | Use |
| --- | --- | --- | --- |
| Default | FlashAttention-3, 3xTF32 video VAE | Calibrated acceptance | Serving |
| Reference | `H3_FLASH3=0 H3_VAE_3XTF32=0` | Bit for bit, original acceptance | Correctness oracle |

## Method

1. Profile before changing code. `tests/serving/profile/` holds a CUPTI
   injection tracer (`cupti_trace.c`, every kernel/copy, optional host API
   calls), `report.py` (time per phase and kernel category) and
   `host_gaps.py` (GPU idle time charged to host API calls).
2. One experiment per Slurm job, one H200 each; independent experiments run
   in parallel on different GPUs, A/B comparisons inside one job.
3. Gates:
   - Changes that must not alter values: `tests/serving/bitcheck.sbatch`, all
     dumps bit-identical to the job-14416 fresh outputs (reference mode).
   - Changes that round differently: microbenchmark error against FP32/FP64,
     decoder acceptance on identical latents, and
     `tests/pytorch/calibrated_accept.py` (below).
   - `tests/serving/serve.sbatch` runs everything for a release candidate.

## Profile of the starting point (jobs 14422, 14423, 14425)

10 s video, 205 s: denoise 132.6 s, video VAE 70.7 s, everything else < 2 s.

| Where | Time | Finding |
| --- | --- | --- |
| DiT attention (FlashAttention-2) | 92.2 s | 70% of denoise; FA2 at about 275-305 TFLOPS, normal for FA2 on Hopper |
| DiT GEMMs (cuBLAS BF16) | 28.8 s | About 715 TFLOPS; little left |
| DiT elementwise kernels | 6.5 s | qkv_rope, swiglu, gate, adaln |
| Video VAE GEMMs | 50.5 s | FP32 SIMT at 44 TFLOPS: no tensor cores |
| Video VAE attention | about 6 s | Unfused GEMM + softmax + pack, FP32 |
| GPU idle | 5 s denoise, 13 s VAE | `cudaFree` synchronizes the device; about 3000 per request |

Correction: an early estimate said FA2 ran at 206 TFLOPS. It used the model
width 5376 instead of the attention width 56 x 128 = 7168.

## Experiments

| # | Change | Result | Verdict | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Stream-ordered memory pool (`cudaMallocAsync`/`cudaFreeAsync`, release threshold kept) instead of `cudaMalloc`/`cudaFree` | 2.3 s video 33.0 -> 28.5 s; 10 s video saves about 8 s; peak memory unchanged | Worked, bit-exact | 14426-bitcheck |
| 2 | `h3_gpu_dispatch_3d`: taller blocks when a grid exceeds 65535 in y or z | Audio longer than about 8.2 s no longer fails ("invalid configuration argument") | Fixed, bit-exact for valid grids | 14418 (failure), 14419 |
| 3 | Remove `-DUNFUSE_FMA` from FlashAttention-2 | Not run: it changes one multiply in the softmax, PyTorch sets it too | Rejected by analysis | - |
| 4 | FlashAttention-3 (flash-attention v2.8.3 `hopper/`) for non-causal BF16 d=128 attention | 2.3-2.7x faster attention (700-750 TFLOPS); same error against an FP32 reference as FA2 (relative L2 2.2e-3) | Worked | 14427-attention |
| 4a | FA3 end to end | 5 s 81.7 -> 62.2 s, 10 s 201.3 -> 140.9 s (with #1) | Worked | 14429-duration |
| 4b | FA3 against the original trajectory acceptance | Fails: final latents move 5-30% | Expected, see calibration | 14428 log, calibration/fa3-14434.json |
| 4c | Calibration: PyTorch math SDPA (more accurate) against PyTorch FA2 | Also fails all six cases: final latents move 4-31% | The original trajectory limits only pass for the identical kernel | calibration/math-sdpa-14433.json |
| 5 | cuBLAS 1xTF32 for the video VAE | 8x faster GEMMs, error 3e-4 (370x FP32) | Rejected: lossy | 14432-vae-gemm |
| 6 | CUTLASS 3xTF32 (sm80 `mma.sync`, `OpMultiplyAddFastF32`) | More accurate than FP32 SIMT (3e-7) but slower: 31 vs 44 TFLOPS | Rejected: slower on Hopper | 14431 log |
| 7 | 3xTF32 from three Hopper TF32 cuBLAS products, weights split at load | 2.3-2.5x faster GEMMs (110 TFLOPS); GEMM error 4.8e-6 | Worked | 14432-vae-gemm |
| 7a | #7 in the video VAE | Decoded error against PyTorch about 1e-4 (FP32 path 1e-6); 99.7% of 8-bit pixels identical, the rest differ by 1; +10 GB | Worked | 14435-vae-3xtf32 |
| 7b | #7 end to end | 5 s 62.2 -> 51.0 s, 10 s 140.9 -> 118.7 s; peak 114 GB | Worked | 14436-duration |
| 8 | Measure SM clock and power every 100 ms, FA3 and FA2 on one GPU | FA3: median 690 W, SM 1530 MHz, power-cap throttling; FA2: 633 W, 1920 MHz. FA3 still wins (10 s denoise 128 -> 78 s) | Finding: the default mode is power-bound | 14445 log |
| 11 | Video VAE tile pipeline: the main thread queues tile k+1 while a host thread unpacks, stitches, blends and finalizes tile k; output patches go to page-locked memory on a separate copy stream, two output slots alternate; latent upload is asynchronous; the suffix token stays resident | 10 s video VAE GPU idle 4.47 -> 1.66 s (all 7 ms per-tile and 87 ms per-chunk gaps gone, job 14462 vs 14441) | Worked, bit-exact both modes | 14462-profile-10s (bitcheck 14454, 14455) |
| 11a | 3xTF32 fusions, same operation order: RMSNorm and SwiGLU write the TF32 high/low parts of the next product's input; each bias is added (`__fadd_rn`) where its product is consumed (QK norm/RoPE, SwiGLU, residual) instead of `bias_f32` | 10 s: `bias_f32` 1.27 s and 23k of 30k `split_tf32` calls removed | Worked, bit-exact (`--fmad=false`) | 14462, 14471 |
| 11b | First fused QK norm/RoPE kernel kept the original thread layout (every thread repeats both 64-term sums) | 222 us per call vs 135 us + 38 us bias before: +0.6 s at 10 s | Did not work; replaced by 11c | 14462-profile-10s |
| 11c | QK norm/RoPE with one block per row and 4 heads: each element loaded once into shared memory, one thread per sum in the original order | 56 us per call. VAE elementwise kernels 4.0 -> 1.5 s at 10 s | Worked, bit-exact | 14471-profile-10s |
| 11d | Stream frames to FFmpeg during the decode: FFmpeg starts before the video VAE, the host thread converts each finished chunk to RGB24 and queues it; same bytes, same order | 10 s: FFmpeg phase 1.03 -> 0.23 s, tail after the last tile 1.17 -> 0.19 s; MP4s byte-identical | Worked | 14471-profile-10s (bitcheck 14468, 14469) |
| 11e | Start FFmpeg on a helper thread (the synchronous start cost 70-80 ms of GPU idle before the first tile) and stitch each tile as it arrives | First-tile gap 70 ms -> 0.3 ms, tail 0.19 -> 0.15 s | Worked, bit-exact | 14476-profile-10s, 14477-profile-short, 14473/14474-bitcheck |

Experiment 11 overall (jobs 14441/14442 before, 14476/14477 after; denoise
varies by about 1.5 s between runs at the power cap, so compare the rest):

| Request | Video VAE wall / busy (s) | FFmpeg (s) | Request minus denoise (s) | Request (s) |
| --- | --- | --- | --- | --- |
| 2.3 s video | 8.49 / 7.47 -> 7.16 / 6.85 | 0.41 -> 0.21 | 9.07 -> 7.53 | 20.7 -> 19.1 |
| 5 s video | 19.66 / 17.42 -> 16.49 / 15.99 | 0.68 -> 0.25 | 20.55 -> 16.99 | 50.9 -> 47.1 |
| 10 s video | 39.21 / 34.75 -> 32.99 / 32.10 | 1.03 -> 0.19 | 40.7 -> 33.7 | 119.4 -> 113.2 |

What is left in the 10 s video VAE: 0.74 s of launch gaps under 5 us
between 357k kernels (most are the unfused FP32 attention: 8 chunks of GEMM,
softmax, GEMM per layer) and a 0.15 s tail (last tile, final blend and RGB24
conversion). `tests/serving/profile/gap_histogram.py` sorts idle time by gap
length. The audio VAE (0.27 s at 10 s) runs before the video VAE in its own
CUDA context; overlapping the two was not tried.

Why 3xTF32 is not as accurate as FP32 here: the Hopper tensor-core FP32
accumulator is not a full IEEE accumulator; the error grows with K (K=8192
gives 1.9e-5). It is still 1000x below BF16 resolution.

## Calibrated acceptance

The 20-step sampler amplifies any rounding change. One forward pass differs
by about 1% (both FA3 and PyTorch math SDPA); after 20 steps the final latents
differ by 4-31%. Bit-exact agreement is only possible with the identical
kernel, so `acceptance.json`'s independent-trajectory limits cannot judge a
different but equally accurate kernel.

`tests/pytorch/calibrated_accept.py` keeps the single-forward limits
(`dit_velocity`: relative L2 0.03, normalized max 0.1 at step 0) and bounds the
final-latent drift over the six required cases by 1.5x the math-SDPA drift
(mean and maximum). The factor covers sampling noise of six chaotic cases.
The limits come from the math-SDPA baseline; the FA3 run (fa3-14434.json)
passes them. Visual check: `/home/homo/model/h3-samples/compare/`, user review
2026-10-10: "挺好的" (good).

## Open items and ideas

- The default mode runs at the 700 W power cap (experiment 8), so all
  kernels run about 20% below the maximum clock. Changes that cut memory
  traffic or energy (fusion, vectorized elementwise kernels) may gain more
  than their kernel time suggests.

- FA3 exits the process on a launch error (`CHECK_CUDA` in upstream
  `cuda_check.h`). Inputs are checked before the call; replace with an error
  return before long-running serving.
- DiT elementwise fusion: about 6.5 s of 132 s at 10 s; can be bit-exact if
  the operation order is kept.
- Video VAE attention: fused FP32 attention instead of GEMM + softmax + pack
  (about 6 s at 10 s).
- Video VAE GPU idle and MP4 overlap: done in experiment 11 (97% busy).
- Lossy options (not started): step reuse, fewer layers, token reduction,
  FP8 GEMMs, sparse attention; multi-GPU sequence parallelism.
