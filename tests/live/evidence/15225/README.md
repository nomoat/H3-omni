# Ten-request incremental session, 2026-10-12

Job 15225 completed with exit 0 in 4m18s on gpu-h200x8-0, using four H200s,
32 CPUs and a 480 GiB allocation. The job has ended and released its GPUs.
Runtime base is 29e9040 plus the session and harness changes in the source
overlay. `source.sha256` records the exact build-cache and overlay archives.
Model export and adapter match the prior 1/2/4-GPU validation in this directory.

The test used a fixed woman-at-a-desk prompt, seed 42, 864x480, 24 fps,
three student denoising steps and one clean KV commit per chunk. It generated
10 sequential requests with 40 clean commits. Each request's audio teacher
ran only after the previous request's video was emitted. Final output:
1,195 frames / 49.7916667 seconds, stereo 32 kHz, 1,593,333 PCM samples.

Checks:

- The first two incremental requests exactly match the existing 243-frame
  path: all 2,799,360 video floats and 25,920 audio floats are bit-identical.
- Each request decodes the known prefix with a one-second output holdback.
  Against every later decode, every previously emitted RGB byte is identical.
  The largest previous-PCM difference is 3.06963921e-6; request 10's RMS
  difference is 5.63764086e-8. These are numerical checks, not listening tests.
- The combined video is encoded once from the emitted RGB/PCM. It does not
  concatenate independently AAC-encoded MP4 streams. All 12 MP4 files
  (10 parts, combined, legacy baseline) decode without FFmpeg errors.
- Local copies of all 12 MP4 files match their recorded SHA-256 hashes.
- The host schedule/layout/normalization contract test also passed locally
  with `-DH3_CUDA` for the portable host implementation.

Visual inspection covered all nine request seams and all nine output-part
seams, using frames at -6, -1, 0 and +6 around each boundary. The same woman,
clothes, desk, flowers, shelves and window persist. No obvious identity swap,
scene cut, or pose discontinuity was visible at these sampled boundaries.
Expressions and head positions change across adjacent frames. This is a
sampled visual check, not a frame-by-frame motion or lip-sync evaluation.

At request seams, mean absolute RGB change is 1.4184 to 2.8092 on a 0-255
scale; the full clip's ordinary adjacent-frame median is 1.3583 and p95 is
2.6980. These values support the visual observation but do not prove semantic
continuity. Every inspected seam's PCM sample jump is below the surrounding
100 ms window's p95 sample jump. No listening, speech-transcription or
speaker-identity evaluation was performed. The prompt is repeated unchanged;
this experiment does not validate an evolving conversation or new prompts.

This is a correctness test with repeated prefix decoding, whose cost grows
with the session. It does not establish bounded-memory, sustained real-time
serving. KV retention itself remains bounded to a video sink plus two chunks.
The ordinary `h3_generate`/HTTP paths remain independent clip generators.
The 12-request audio-history reset is outside this ten-request test.

Files:

- `continuity.jsonl`: exact comparisons, request ranges, prior-media stability.
- `generation.log`: every clean KV commit and request return.
- `seam-metrics.json`: RGB and PCM diagnostics at both kinds of boundary.
- `mp4.sha256`, `source.sha256`, `gpus.csv`, `slurm.txt`: provenance.
- Local videos, latent dumps and six inspection sheets:
  `outputs/live/continuity-15225/` (gitignored, in the main checkout).
- Remote artifacts: `/home/cj/h3-live/results/15225/`.
- Raw emitted RGB/PCM on the compute node: `/var/tmp/h3-continuity-15225/`.
