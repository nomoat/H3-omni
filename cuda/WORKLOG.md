# Serving latency work log

Goal: lower the latency of one warm serving request on one H200. "Lossless"
first: no fewer steps, layers or tokens, and no approximation of the model.

All times are warm requests in one resident process at 864x480, 20 steps,
unless stated otherwise. Evidence paths are relative to `tests/serving/evidence/`
(compact copies) and `/home/cj/h3-serve/results/<job>` on the cluster (full).

## Current result

| Request | Start (job 14419) | Experiments 1-8 (job 14436) | Experiments 9-11 merged (job 14480) | FP16 video VAE, #13 (job 14734) | VAE kernels and pipeline, #14 (job 14771) | vLLM-Omni, 1 GPU | Change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.3 s video (56 frames) | 33.0 s (job 14416) | 20.8 s (job 14438) | about 19 s ¹ | not measured | not measured | - | - |
| 5 s video (124 frames) | 81.7 s | 51.0 s | 41.0 s | 32.4 s | 32.0 s | 33.0 s | -61% |
| 10 s video (243 frames) | 201.3 s | 118.7 s | 98.5 s | 81.2 s | 80.4 s | 82.9 s | -60% |
| 15 s video (362 frames) | - | - | 174.3 s (job 14488) | not measured | not measured | - | - |

¹ Agent profiles of #9-#11 separately (jobs 14458, 14477); not remeasured
after the merge.

Peak GPU memory in resident serving: 100889 MiB at 10 s after #14 (job
14771, nvidia-smi every 0.5 s); before #13: 114 GB at 10 s, 118 GB for 1344x768,
768x1344 and 15 s in one process (job 14488); the H200 has 141 GB.

Two modes:

| Mode | Settings | Matches PyTorch | Use |
| --- | --- | --- | --- |
| Default | FlashAttention-3, FP16 video VAE (#13) | Calibrated acceptance; video VAE no worse than the diffusers FP16 decode | Serving |
| Accurate VAE | default + `H3_VAE_PRECISION=3xtf32` (the default before #13) | Calibrated acceptance | Serving when the decode must stay within 1e-4 of FP32 |
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
| 9 | Hand-written DiT elementwise kernels (`cuda/fast_primitives.cuh`): 8/16-byte vector loads and stores, AdaLN with one 128-thread block per row and no unused second reduction, the gate fused into the following AdaLN (gated row kept in registers), one warp per (row, head) for q/k RMS + RoPE. Same per-element operations, rounding points and reduction order as the generated kernels | Microbenchmark at 29700 rows, unthrottled clock: AdaLN 0.524 -> 0.236 ms, gate 0.617 -> 0.225 ms, gate + AdaLN 1.142 -> 0.393 ms (fused), SwiGLU 1.432 -> 0.647 ms (3.95 TB/s), qkv_rope 1.983 -> 0.841 ms. All outputs bit-identical on four shapes, including in-place gate and odd tails | Worked | 14450-elementwise |
| 9a | First qkv_rope version: RoPE partner value re-read with scalar loads, as in the generated kernel | 1.983 -> 1.325 ms only (1.9 TB/s). The partner value bf16(x[pair] * inverse * w[pair]) is the partner lane's own rounded value, so #9 takes it with a warp shuffle (0.841 ms) | Replaced by #9 | 14447-elementwise-v1 |
| 9b | #9 bit-exactness gates | Reference mode (against 14416) and default mode (against 14438): all 7 requests bit-identical | Passed | 14451-bitcheck-reference, 14452-bitcheck-default |
| 9c | #9 CUPTI profile, 10 s and short sets | 10 s denoise elementwise kernels 7.21 -> 2.47 s (qkv_rope 2.69 -> 0.94, SwiGLU 1.79 -> 0.78, gate 1.56 + AdaLN 1.17 -> fused 0.75 s); request 119.4 -> 116.8 s. 2.3 s request 20.7 -> 19.9 s, 5 s 50.9 -> 49.3 s. Baselines 14441/14442 ran on another node (attention and GEMM 1-3% slower on this one), so see 9d for the clean comparison | Worked | 14457-profile-10s, 14458-profile-short |
| 9d | Same-GPU A/B, generated (`H3_GENERATED_ELEMENTWISE=1`) against #9, two warm 10 s requests each, clock sampled in denoise | Request 119.2 -> 116.2 s (-2.5%), denoise 78.51 -> 75.49 s. Denoise SM clock median 1515 -> 1485 MHz at 691 W: the clock did not rise, the faster memory-bound kernels draw the same power cap | Worked (-3.0 s); clock-gain hypothesis not confirmed | 14465-elementwise-ab |
| 10 | Video-VAE attention as one fused kernel (`cuda/vae_attention.cuh`): 3xTF32 `mma.sync` m16n8k8 for Q K^T and P V, online FP32 softmax (`expf`), token-major input (no pack/unpack); first version keeps O in the tensor-core accumulator | 1797 tokens x 32 heads: 1.91 -> 0.63 ms per layer (3.0x); error against FP64 3e-6 to 1.3e-5, FP32 path 0.8e-6 to 1.2e-6; worst when attention is nearly uniform | Rejected: 10x less accurate than FP32 | 14446-vae-attn-first |
| 10a | #10 plus promotion: each 64-key block's P V goes into a fresh accumulator, added to O with FP32 adds | Error 5.6e-7 to 1.3e-6, the same as the FP32 path; 0.78 ms (255 registers). A fourth product (xl*yl) without promotion: same error, 0.92 ms | Worked (accuracy) | 14449-vae-attn-promote |
| 10b | 128-bit fragment layout in shared memory (one load = high and low parts of both B registers), scalar global loads, split in the kernel; 4 and 8 warps | 0.87 ms (4 warps), 0.71 ms (8 warps): slower than #10a; staging the K/V block dominates | Rejected | 14453-vae-attn-layout |
| 10c | K and V split once per call into fragment-ordered high/low buffers (`split_kv`, 61 MB transient), `cp.async` double buffer, 8 warps x 16 rows | 0.70 ms including the split (2.7x); same error as #10a | Worked | 14459-vae-attn-bench-v5 |
| 10d | #10c in the video VAE (default mode only, `H3_VAE_3XTF32`), decoder suite + RGB on identical latents | Decoder suite passes; video relative L2 against PyTorch 1.4985e-4 / 1.4011e-4 / 6.81e-5 / 5.20e-5 / 7.09e-5 (#7a: 1.4987e-4 / 1.4013e-4 / 6.81e-5 / 5.20e-5 / 7.09e-5); RGB against the FP32 decoder, 4 requests: relative L2 2.05e-5 to 8.08e-5, 99.687% to 99.826% of 8-bit pixels identical, the rest off by 1 (#7a: same to 3 digits) | Worked: no measurable loss against #7a | 14460-vae-attention-accuracy |
| 10e | #10c end to end (10 s, CUPTI) | Video VAE 39.2 -> 29.7 s; attention 14.2 s (2 SGEMMs 8.95, softmax 3.80, pack/unpack 1.03, gemv 0.46) -> 5.2 s; VAE kernel launches 409,500 -> 212,940; request 119.4 -> 110.7 s | Worked | 14461-profile-10s-v5 |
| 10f | Interleave the 8 independent MMA chains (Q K^T: kk outer; P V: key group outer, P split one group at a time); each accumulator keeps its summation order | 0.548 ms (3.5x the FP32 path, 48 effective TFLOPS); output bit-identical to #10c in all 8 bench cases | Worked | 14464-vae-attn-bench |
| 10g | #10f end to end | Video VAE 28.8 s (baseline 39.2 s); attention 4.09 s (kernel 3.91 + split 0.18); request 107.4 s (baseline 119.4 s; this node's denoise was 1.6 s faster, so about 10.4 s is from this change); reference mode bit-exact, 7 of 7 requests; decoder suite and RGB numbers identical to #10d | Worked | 14467-profile-10s, 14466-bitcheck, 14470-vae-attention-accuracy-final |
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
| 12 | Merge #9, #10, #11. Git merged #10 into the unfused block path, which the default mode no longer uses; fixed by hand to select the fused attention in #11's 3xTF32 path | Default latents and audio bit-identical to 14438, RGB 99.996% pixels identical, rest +-1 (14478); reference mode bit-identical to 14416 (14481); full acceptance incl. calibrated passes (14479); 5 s 41.0 s, 10 s 98.5 s (14480) | Worked | 14478-mergecheck, 14479-merged-accept, 14480-duration, 14481-bitcheck-reference |
| 12a | Differential test outside the serving set, old (14438 source) vs merged build, both modes: 1344x768, 768x1344, 15 s, first/last frame | Reference mode bit-identical in all four; default latents and audio bit-identical, RGB at most 1 level off (99.997%+ identical) | Passed | 14482-14485-scenario |
| 12b | Ref2VA image and reference video, first try | Not testable: the cluster had only the FL2VA checkpoint ("ordered references require the Ref2VA checkpoint"); old and new builds fail the same way | Environment | 14486-14487-scenario |
| 12d | Download Ref2VA (MiniMaxAI/MiniMax-H3 42ed227e). First only the transformer (66 GB): h3 also reads Ref2VA/tokenizer, text_encoder and VAEs and failed with an empty-JSON tokenizer error. Then the full 144 GB; all 29 large files match the Hugging Face sha256. Only the transformer differs from FL2VA | Complete checkpoint in the shared model dir | Done | 14489-14490 (failed runs) |
| 12e | Ref2VA needs FFprobe for reference inputs ("cannot start FFprobe"); none on the cluster. Fetched the johnvansickle 7.0.2 static release, whose ffmpeg is byte-identical to the one in use, and set `H3_FFPROBE` | Ref2VA runs | Done | 14491-14492 (failed runs) |
| 12g | Clear error for an incomplete Ref2VA/: h3 checks Ref2VA tokenizer, text encoder and VAEs at load; prompt-only requests still run, ordered references name the missing file | Partial directory test passes; reference mode bit-identical to 14416 | Done | 14496-ref2va-partial, 14497-bitcheck-reference |
| 13 | FP16 video VAE (`cuda/vae_fp16.cuh`, default; `H3_VAE_PRECISION=fp16\|3xtf32\|fp32`): block weights rounded to FP16 at load; RMSNorm (FP32 statistics) writes FP16 rows; QKV, output, w1, w2 as FP16 cuBLAS products with FP32 accumulation and FP32 output; bias, QK norm, RoPE, SwiGLU and residual in FP32, Q/K/V and SwiGLU output rounded to FP16; FlashAttention-3 FP16, head dim 64 (second instantiation in `flash3_attention.cu`). Input projection, output LayerNorm and projection stay FP32. Same recipe as the diffusers mixed-precision decode (`_keep_in_fp32_modules`), except that products return FP32 instead of FP16 | Error against the PyTorch FP32 decode on 9 cases (6 acceptance trajectories, 3 served requests up to 864x480x56): relative L2 3.9e-5 to 2.6e-4, 98.5% to 99.7% of 8-bit pixels identical, never off by more than 1. diffusers FP16: 3.2e-4 to 1.6e-3, up to 8 levels off; BF16: 2.5e-3 to 1.3e-2, up to 43 levels; FP16 autocast (vLLM-Omni) equals diffusers FP16. 3xTF32: 2.1e-5 to 1.5e-4. Video VAE + MP4 at 10 s: about 23 s -> 6.2 s. 5 s 41.0 -> 32.4 s, 10 s 98.5 -> 81.2 s. Reference mode bit-identical to 14416; 3xTF32 mode gives the same latents and audio as 14438 and the same RGB as the merged build (14478) | Worked: faster than vLLM-Omni on one GPU, 6x to 10x less error than the diffusers FP16 decode | 14732-vae-precision, 14734-duration, 14735-bitcheck-reference, 14736-bitcheck-3xtf32 |
| 14 | Video VAE after #13, from a kernel profile (`tests/pytorch/vae_profile.py`, torch.profiler around the resident decoder, 10 s latent 67x30x54). Profile before: 4.92 s decode, GEMMs 2.49 s, FA3 0.56 s, SwiGLU 0.43 s, QKV/RoPE 0.41 s, residual + RMSNorm 0.40 s, FP32 SIMT output projection 0.13 s, and 12 GPU stalls of 21-34 ms (0.31 s) at the temporal-chunk boundaries: the host thread blends and finalizes each chunk, and with 2 pipeline slots the GPU had nothing queued. A concurrent libx264 encode cost only 0.06 s (14767). Changes: 4 pipeline slots; FP16 mode only: residual add fused with the following RMSNorm (norm2, and the next block's norm1 when resident; same arithmetic and reduction order), QKV/RoPE with one warp per (row, head) and shuffle reductions, SwiGLU with float4 loads, output projection as 3xTF32 | Decode 4.92 -> 4.26 s (QKV/RoPE 411 -> 141 ms, SwiGLU 428 -> 277 ms, residual + norm 396 -> 273 ms, projection 132 -> about 55 ms). Serving: 10 s video VAE + MP4 6.04 -> 5.22 s, 5 s 32.4 -> 32.0 s, 10 s 81.2 -> 80.4 s. Precision, 9 cases: all pass, h3-fp16 relative L2 4.8e-5 to 2.6e-4, 98.4% to 99.5% of 8-bit pixels identical, never more than 1 level off. GPU at the 700 W cap during the VAE (SM 1.6-1.9 GHz, reason 0x4). Reference mode bit-identical to 14416 | Worked | 14766-vae-profile-before, 14769-vae-profile, 14770-vae-precision, 14771-duration, 14774-bitcheck-reference |
| 15 | Merge FastH3 (`VSA=1`, commit 299e8a6) and #13-#14 into `cuda-h200-phase1` (2c1a59f); one branch from here on | Reference mode bit-identical to 14416 (7/7). Default serving unchanged: 5 s 31.9-32.1 s, 10 s 80.2-80.6 s. FastH3 512x512 four-step parity 20/20 pass, DiT bit-identical; decoders pass (video relative L2 1.7e-4 with the FP16 VAE). FastH3 serving, 864x480: 5 s 10.5-11.6 s, 10 s 21.0-22.5 s (5 s was 22.5 s in 14583); denoise 5.1 / 10.8 s, video VAE 2.5 / 4.9 s, but the MP4 tail is 2.4-6.3 s: libx264 does not keep up with this content during the decode | Merged and verified | 14777-bitcheck-reference, 14778-duration, tests/fasth3/evidence/14779 |
| 16 | Multi-GPU (`--gpus N` / `H3_GPUS`): one process, one thread and one model copy per GPU, NCCL 2.25 from the container. DiT: Ulysses sequence parallelism, all-to-all of Q/K/V and attention output around each attention, all-gather of the hidden rows after the blocks; packing, final heads and the sampler replicated. Video VAE: tiles round-robin over GPUs, per-GPU unpack threads, in-order stitching; then the stitch, temporal blend, normalization and RGB24 passes split over 8 host threads | Bit-identical to one GPU at N = 2, 4, 7 (two requests each). 2 GPUs: 5 s 18.1-18.3 s, 10 s 44.2-44.7 s (vLLM-Omni 18.9 / 45.4). 4 GPUs: 5 s 9.2-9.4 s, 10 s 22.2-22.4 s (vLLM-Omni 10.2 / 24.0); 10 s denoise 20.4 s (vLLM-Omni 21.6), video VAE 1.27 s. 7 GPUs, before the host passes: 6.5-6.8 / 15.2-15.5 s with a 2.4 s video VAE. Reference mode on one GPU still bit-identical to 14416 8 GPUs (14792): bit-identical, 5 s 5.4-5.7 s, 10 s 12.6-13.1 s (vLLM-Omni 6.1 / 13.6) | Worked: faster than vLLM-Omni at 2, 4 and 8 GPUs | 14780-multigpu-2, 14781-multigpu-2, 14783-multigpu-4, 14785-multigpu-7, 14789-multigpu-4, 14784/14791-bitcheck-reference |
| 12f | Ref2VA image and reference video, old vs merged build, both modes | Reference mode bit-identical; default latents and audio bit-identical, RGB 99.998% identical, rest +-1; peak 40-41 GB one-shot | Passed | 14493-14494-scenario |
| 12c | Resident serving at the largest shapes | 1344x768 44.1 s, 768x1344 44.4 s, 15 s 174.3 s; peak 118 GB of 141 GB | Passed | 14488-resident-large |

Why 3xTF32 is not as accurate as FP32 here: the Hopper tensor-core FP32
accumulator is not a full IEEE accumulator; the error grows with K (K=8192
gives 1.9e-5). It is still 1000x below BF16 resolution.

The same accumulator explains #10 against #10a. Attention sums 1797 products
P V whose sum is much smaller than the sum of their sizes when attention is
nearly uniform, so accumulator rounding is amplified. Summing each 64-key
block in a fresh accumulator and adding the blocks in FP32 removes the
difference at no cost after #10f (0.534 ms without, 0.548 ms with).

Video-VAE attention kernel limits (#10f): 255 registers per thread, so one
CTA of 8 warps per SM; plain TF32 in the same kernel takes 0.40 ms, so most of
the time is softmax, splitting and latency, not tensor-core work. Next steps if
needed: WGMMA (sm_90a, B from shared memory, V transposed), or `exp2` with a
folded log2(e) (changes rounding of every probability; measure first).

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

- The default mode runs at the 700 W power cap (experiment 8), so kernels run
  about 20% below the maximum clock. Faster elementwise kernels (9d) did not
  raise the clock; denoise power stays at the cap.
- FA3 exits the process on a launch error (`CHECK_CUDA` in upstream
  `cuda_check.h`). Inputs are checked before the call; replace with an error
  return before long-running serving.
- DiT elementwise: 2.5 s left at 10 s (after #9). More needs fusion into GEMM
  epilogues (gate into the output projections, SwiGLU into fc1), which changes
  the GEMM path. About 0.4 s more may be left in qkv_rope and gate + AdaLN.
- Video VAE attention: done in the default mode (#10). The reference mode
  still uses GEMM + softmax + pack.
- Video VAE idle time and MP4 overlap: done (#11, 97% busy). The audio VAE
  (0.27 s at 10 s) could overlap the video VAE.
- Video VAE after #14: 5.2 s with MP4 at 10 s (decode 4.26 s in isolation).
  GEMMs are now 60% of the decode at about 680 TFLOPS. FP16 product outputs
  would cut the SwiGLU and QKV traffic, but at diffusers-FP16 error; not
  tried. The audio VAE (0.25 s) could still overlap the video VAE.
- FastH3: the MP4 tail (libx264 after the decode) is now the largest part
  after denoise, 2.4-6.3 s. Options: more encoder threads, a faster preset,
  or NVENC.
- Multi-GPU (#16): 8 GPUs (job 14792) bit-identical; 5 s 5.4-5.7 s, 10 s
  12.6-13.1 s (vLLM-Omni 6.1 / 13.6 s), denoise 10.9 s, video VAE 1.15 s.
  Each GPU holds a full DiT copy and the
  activations of the whole sequence although it uses 1/N of the rows.
- Lossy options (not started): step reuse, fewer layers, token reduction,
  FP8 GEMMs, sparse attention; multi-GPU sequence parallelism.
