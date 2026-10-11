#!/bin/bash
#SBATCH --job-name=h3-vae-precision
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cj/h3-serve/vae-precision-%j.log
# Video-VAE precision study (tests/pytorch/vae_precision.py): the six
# acceptance trajectories and three served requests (864x480 and 512x512,
# 56 and 22 frames) from job 14438, decoded by PyTorch in FP32, FP16, BF16
# and FP16 autocast and by h3 in FP32, 3xTF32 and FP16.
# Usage: sbatch vae_precision.sbatch SOURCE_TAR
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/cj/h3-serve
SOURCE=$1
STAGE=/var/tmp/h3-vae-precision-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/python" "$STAGE/diffusers" "$STAGE/.deps" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
tar -xf /home/cj/h3-cuda/python-deps.tar -C "$STAGE/python"
tar -xf /home/cj/h3-cuda/diffusers.tar -C "$STAGE/diffusers"
WEIGHTS=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA
echo "node $(hostname)"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$WEIGHTS" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/libh3_model.so"
TRAJ=/home/cj/h3-serve/results/14479/accept/trajectories
FRESH=/home/cj/h3-serve/results/14438/fresh
ARGS=()
for case in 256x256x22-seed0 256x256x22-seed1 256x256x22-seed42 512x512x22-seed0 512x512x22-seed1 512x512x22-seed42; do
    ARGS+=(--trajectory "$TRAJ/$case/latents.safetensors")
done
ARGS+=(--raw "r1-boat-864x480x56=$FRESH/r1-boat/video_latent.f32:17:30:54")
ARGS+=(--raw "r2-fox-864x480x56=$FRESH/r2-fox/video_latent.f32:17:30:54")
ARGS+=(--raw "r4-cat-512x512x22=$FRESH/r4-cat/video_latent.f32:7:32:32")
mkdir -p results
python3 tests/pytorch/vae_precision.py --weights "$2" --diffusers "$1/diffusers" --output "$1/results" "${ARGS[@]}" \
    2>&1 | grep -v Warning
RUN
