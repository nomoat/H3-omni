#!/bin/bash
#SBATCH --job-name=h3-live-service
#SBATCH --partition=train
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=0
#SBATCH --output=/home/cj/h3-live-service/logs/service-%j.log
# Bounded-session playground pilot. Keep resident between requests.
# Usage: sbatch --export=ALL h3-live.sbatch BUILD_CACHE_TAR COMMITTED_OVERLAY_TAR
# GPU count follows the Slurm allocation, including explicit --gres overrides.
set -euo pipefail
LIVE_GPUS=${SLURM_GPUS_ON_NODE:?Slurm GPU allocation is required}
case "$LIVE_GPUS" in 1|2|4|8) ;; *) echo "Unsupported GPU allocation: $LIVE_GPUS" >&2; exit 1 ;; esac
LIVE_ROOT=/home/cj/h3-live-service
LIVE_STAGE=/var/tmp/h3-live-15182
mkdir -p "$LIVE_STAGE" "$LIVE_ROOT/logs"
tar -xf "$1" -C "$LIVE_STAGE"
tar -xf "$2" -C "$LIVE_STAGE"
sha256sum "$1" "$2" > "$LIVE_ROOT/source-$SLURM_JOB_ID.sha256"
test -s "$LIVE_ROOT/secrets/key"
trap 'if [ "$(cat "$LIVE_ROOT/job-id" 2>/dev/null)" = "$SLURM_JOB_ID" ]; then rm -f "$LIVE_ROOT/endpoint"; fi' EXIT
apptainer exec --nv --bind /data0:/data0,/var/tmp:/var/tmp,/home/cj:/home/cj \
 /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$LIVE_STAGE" "$LIVE_ROOT" "$LIVE_GPUS" <<'RUN'
set -euo pipefail
cd "$1"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 H3_GPUS="$3" VSA=1
make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_worker"
echo "$SLURM_JOB_ID" > "$2/job-id"
echo "$(hostname):18961" > "$2/endpoint.tmp"
mv "$2/endpoint.tmp" "$2/endpoint"
while true; do
 python3 serve/live_server.py --worker build-cuda/h3_worker \
   --model /data0/models/h3-taomate/native --gpus "$H3_GPUS" --port 18961 \
   --key-file "$2/secrets/key" --log "$2/logs/worker-$SLURM_JOB_ID.log" || true
 sleep 5
done
RUN
