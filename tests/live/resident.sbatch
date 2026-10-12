#!/bin/bash
#SBATCH --job-name=h3-live-resident
#SBATCH --partition=h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cj/h3-live/resident-%j.log
# Usage: sbatch resident.sbatch SOURCE_TAR GPU_COUNT [EXISTING_STAGE]
set -euo pipefail
LIVE_STAGE=${3:-/var/tmp/h3-live-$SLURM_JOB_ID}
LIVE_RESULT=/home/cj/h3-live/results/$SLURM_JOB_ID
mkdir -p "$LIVE_STAGE/.deps" "$LIVE_STAGE/icu" "$LIVE_RESULT"
tar -xf "$1" -C "$LIVE_STAGE"
if [ ! -f "$LIVE_STAGE/.deps/flash3-hopper/flash.h" ]; then
 tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$LIVE_STAGE/.deps"
 tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$LIVE_STAGE/.deps"
 tar -xf /home/cj/h3-cuda/icu.tar -C "$LIVE_STAGE/icu"
fi
if [ "${VSA:-0}" = 1 ] && [ ! -d "$LIVE_STAGE/.deps/thunderkittens" ]; then
 tar -xzf /home/cj/h3-fasth3/tk-deps.tar.gz -C "$LIVE_STAGE/.deps"
fi
printf '%s\n' "$LIVE_STAGE" > "$LIVE_RESULT/stage.txt"
apptainer exec --nv --bind /data0:/data0,/var/tmp:/var/tmp,/home/cj:/home/cj \
 /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$LIVE_STAGE" "$LIVE_RESULT" "$2" <<'RUN'
set -euo pipefail
cd "$1"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 H3_GPUS="$3"
make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr" all "$1/build-cuda/h3_live_resident"
build-cuda/h3_live_resident /data0/models/h3-taomate/native "$2" > "$2/timing.jsonl" 2> "$2/generation.log"
sha256sum "$2"/*.mp4 | tee "$2/mp4.sha256"
cmp "$2/full-cold.mp4" "$2/full-warm.mp4"
cmp "$2/full-warm.mp4" "$2/full-host.mp4"
cat "$2/timing.jsonl"
RUN
