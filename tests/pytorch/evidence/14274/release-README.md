# H3 CUDA phase-one runtime

Built and validated on one NVIDIA H200 in Slurm job 14274 (2026-10-10).
The executable uses C/C++ and CUDA libraries. It does not depend on Python,
PyTorch or libtorch at runtime. Original model weights are required separately.

Seven complete text-to-video/audio trajectories passed, including an 82-token
prompt at 864x480 with 56 frames. All use 50 DiT layers and 20 model evaluations.
All 560 per-step comparisons were exact; all 28 float video/audio decoder
comparisons passed the fixed tolerances. Image/keyframe conditioning and Ref2VA
are outside this validation scope. No layer skipping, prediction reuse, token
merging or quantization was used.

Run inside an allocated single-H200 Slurm job on the documented cluster:

```sh
/home/cj/h3-cuda/releases/14274/run.sh \
  -d /data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe \
  -p 'A cat walks through a sunny garden.' \
  --width 256 --height 256 --frames 22 --steps 20 --layers 50 \
  --reuse 1 --core-reuse 1 --seed 0 -o cat.mp4
```

The launcher uses `/data0/apptainer/pytorch_291_cu128.sif` with GPU access and
sets paths for the bundled ICU libraries and FFmpeg. The container supplies
CUDA, cuBLAS and cuDNN. Model loading is included in CLI wall time.

`source.tar` contains the exact build source, tests, reference pins and tolerance
contract. Complete fixtures and reports are in `/home/cj/h3-cuda/results/14274`.
`summary.json` contains the aggregate results. Historical job 14269 is superseded
because its oracle retained duplicate first-step timestep rows. This build uses
the official distinct-row timestep plan and corrected non-square FP64 positions.

From this directory, `sha256sum -c SHA256SUMS` verifies the runtime files, source
archive and launcher. This is a correctness baseline; no speedup is claimed.
