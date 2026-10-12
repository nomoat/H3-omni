# Four-H200 benchmark, 2026-10-12

Job 15202 completed with exit 0 in 4m19s on `gpu-h200x8-0` (h200 partition),
using four H200s, 32 CPU threads and a 480 GiB memory allocation. Model export,
container, precision and native runtime match the preceding validation.
Runtime commit: `1a1f588`. Only the benchmark harness changed.

- `timing.jsonl`: all ten resident generations, including cold cases.
- `summary.json`: 5/10-second warm ranges and medians, phase timings, hashes,
  and comparisons against the two-GPU job 15192. `all_comparisons_equal` is
  **false**; cross-count exact parity did not pass.
- `generation.log`: chunk timings and clean KV commits (14 heads per GPU).
- `mp4.sha256`: all five full-size 10-second outputs are identical, as are
  all four 5-second outputs. The tenth output is the small canvas.
- `projection-probe.jsonl`: independent row/head-splitting diagnostic.
- `media-difference.json`: decoded full-size media and small latent differences.

All ten MP4 files decode without FFmpeg errors. Warm 5-second runs take
4.47-4.63 s; warm 10-second runs take 8.38-8.87 s. These short runs exceed
playback speed, but do not establish sustained live service. The existing
15-second cap, whole-latent decode and independent-call history reset remain.

Cross-count output differences remain open. The isolation probe reproduces
base FC2 GEMM rounding changes after four-way splitting, while all seven
56-to-14-head attention cases are exact. This supports a numerical cause but
is not an end-to-end causal trace or a full quality evaluation.

Build-cache archive SHA-256:
`a2a5baf02db88f2810f8782c58d0eab35531ee74c8fbbfe2e703a8691fbdf3df`.
Benchmark-overlay archive SHA-256:
`56cf3db5d5103589306ce1c9571b0ac74b244e351676873da062b8c348784f61`.
Both archives and full artifacts are under `/home/cj/h3-live/` on the cluster.
Local sample: `outputs/live/taomate-h3-4gpu.mp4` (gitignored).
