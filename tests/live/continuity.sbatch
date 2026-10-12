#!/bin/bash
#SBATCH --job-name=h3-live-continuity
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:40:00
#SBATCH --output=/home/cj/h3-live/continuity-%j.log
# Usage: sbatch --export=ALL continuity.sbatch BUILD_CACHE_TAR OVERLAY_TAR
set -euo pipefail
LIVE_STAGE=/var/tmp/h3-live-15182
LIVE_RESULT=/home/cj/h3-live/results/$SLURM_JOB_ID
LIVE_SCRATCH=/var/tmp/h3-continuity-$SLURM_JOB_ID
mkdir -p "$LIVE_STAGE" "$LIVE_RESULT" "$LIVE_SCRATCH"
tar -xf "$1" -C "$LIVE_STAGE"
tar -xf "$2" -C "$LIVE_STAGE"
sha256sum "$1" "$2" > "$LIVE_RESULT/source.sha256"
apptainer exec --nv --bind /data0:/data0,/var/tmp:/var/tmp,/home/cj:/home/cj \
 /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$LIVE_STAGE" "$LIVE_RESULT" "$LIVE_SCRATCH" <<'RUN'
set -euo pipefail
cd "$1"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 H3_GPUS=4 VSA=1
make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_live_continuity"
nvidia-smi --query-gpu=name,uuid --format=csv > "$2/gpus.csv"
set +e
build-cuda/h3_live_continuity /data0/models/h3-taomate/native "$2" "$3" > "$2/continuity.jsonl" 2> "$2/generation.log"
status=$?
set -e
cat "$2/continuity.jsonl"
sha256sum "$2"/*.mp4 > "$2/mp4.sha256"
if [ -f "$3/published.rgb24" ]; then
 python3 tests/live/continuity_review.py "$3" "$2"
fi
exit "$status"
RUN
