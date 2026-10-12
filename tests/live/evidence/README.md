# TaoMate-H3 validation, 2026-10-12

Native runtime base: `e871c2e` plus the live-mode change in this commit.
Upstream contract: TaoLiveAIGC/TaoMate-H3 commit
`6b2f9984acf681e2f387188f5d1f3b26e504315b`.
Adapter SHA-256:
`a3ede4a1dce7030adc738fd8926ffd090e1b92e7f38d569626a35a553c0409ad`.
Cluster native export: `/data0/models/h3-taomate/native`.
Container: `/data0/apptainer/pytorch_291_cu128.sif`.
GPUs: NVIDIA H200, native CUDA build for sm_90, Hopper FA3 for sm_90a.

The runtime source archive tested by jobs 15191 and 15192 was
`/home/cj/h3-live/source-v5.tar`, SHA-256
`af5f1d8d60d0a35b97bac95f8a3b63d75f1611be5b37b861767ecee76883d912`.
Later changes add test assertions, evidence and documentation; runtime code
is unchanged. `resident.c` records the exact prompt, seed, shapes and cache
settings. Results are individual runs, not a throughput or latency survey.

| Job | Node / GPUs | Check | Result |
| --- | --- | --- | --- |
| 15191 | gpu-h200x1-0 / 1 | Four complete resident audio/video generations | COMPLETED, exit 0 |
| 15192 | gpu-h200x8-3 / 2 | Same four cases with Ulysses and multi-GPU VAE | COMPLETED, exit 0 |
| 15195 | gpu-h200x8-3 / 1 | VSA-enabled build and regression | COMPLETED, exit 0 |

- `comparison.json`: single/two-GPU latent, decode and MP4 comparisons, and timings.
- `primitives.json`: all 69 existing CUDA primitive cases pass.
- `attention.json`: all six rectangular attention cases pass against independent PyTorch math.
- `lora.json`: eight native-versus-released-adapter projections pass; GPU sampler matches FP32 host operation order exactly.
- `sharding.json`: all 21 row/head-splitting checks are exactly equal.
- `*-info.txt` and `mismatch.txt`: original, fast and live checkpoints load, and a requested mode/checkpoint mismatch is rejected.
- `15191-generation.log` and `15192-generation.log`: phase timings and clean cache retention for each of the four full runs.

The CPU host-contract test also passed. All eight MP4 files decode through
FFmpeg without errors. All six 864x480 MP4 files have SHA-256
`7523129a3511f06fdd02936622dd49f9f1d7d23eb040fcc57ef6538276df748c`.
The full video is 243 frames at 24 fps, H.264, with stereo 32 kHz AAC audio.

No end-to-end numerical parity claim is made against the upstream Python
program: its normal RNG and normalization reductions differ. Component
checks, native determinism and visual continuity were validated. Original
and FastH3 received load/primitive regression checks, not a new full-video
benchmark in this change.

Full remote artifacts: `/home/cj/h3-live/results/{15191,15192,15195}`.
Local sample: `outputs/live/taomate-h3-2gpu.mp4` (gitignored).
