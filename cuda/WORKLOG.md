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

- FA3 exits the process on a launch error (`CHECK_CUDA` in upstream
  `cuda_check.h`). Inputs are checked before the call; replace with an error
  return before long-running serving.
- DiT elementwise fusion: about 6.5 s of 132 s at 10 s; can be bit-exact if
  the operation order is kept.
- Video VAE attention: fused FP32 attention instead of GEMM + softmax + pack
  (about 6 s at 10 s).
- Remaining GPU idle time in the video VAE; overlap VAE decode with MP4
  encoding.
- Lossy options (not started): step reuse, fewer layers, token reduction,
  FP8 GEMMs, sparse attention; multi-GPU sequence parallelism.
