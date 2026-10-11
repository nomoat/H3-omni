#!/bin/bash
#SBATCH --job-name=h3-vae-profile
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/home/cj/h3-serve/vae-profile-%j.log
# Kernel profile of the resident video-VAE decoder on a 10 s (67x30x54) latent.
# Usage: MODE=fp16|3xtf32 sbatch vae_profile.sbatch SOURCE_TAR
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/cj/h3-serve
SOURCE=$1
STAGE=/var/tmp/h3-vae-profile-$SLURM_JOB_ID
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
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$WEIGHTS" "${MODE:-fp16}" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/libh3_model.so"
mkdir -p results
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
H3_VAE_PRECISION=$3 python3 tests/pytorch/vae_profile.py --weights "$2" --output "$1/results" 2>&1 | grep -v Warning
RUN
