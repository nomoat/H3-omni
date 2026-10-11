#!/bin/bash
#SBATCH --job-name=h3-video
#SBATCH --partition=h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=0
#SBATCH --output=/home/cj/h3-video/logs/%x-%j.log
# Resident FastH3 video service: builds h3_worker (VSA=1) on node-local disk,
# then runs serve/server.py on all GPUs of the job. Restarts the server if the
# worker exits. Writes "host:port" to $ROOT/endpoint for the rust-build tunnel.
# Usage: sbatch --gres=gpu:N --nodelist=NODE h3-video.sbatch SOURCE_TAR N [PORT]
set -euo pipefail
ROOT=/home/cj/h3-video
SOURCE=$1
N=$2
PORT=${3:-18960}
KEY=$ROOT/secrets/key
MODEL=/data0/models/h3-fasth3/vsa-native
STAGE=/var/tmp/h3-video-$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$STAGE/python" "$ROOT/logs"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-fasth3/tk-deps.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
test -s "$KEY"
echo "node $(hostname) N=$N port=$PORT model=$MODEL"
trap 'rm -f "$ROOT/endpoint"; rm -rf "$STAGE"' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$N" "$PORT" "$KEY" "$ROOT" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"; N="$3"; PORT="$4"; KEY="$5"; ROOT="$6"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
make -f cuda/Makefile -j32 VSA=1 ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_worker"
echo "$(hostname):$PORT" > "$ROOT/endpoint.tmp" && mv "$ROOT/endpoint.tmp" "$ROOT/endpoint"
while true; do
    python3 serve/server.py --worker build-cuda/h3_worker --model "$MODEL" --gpus "$N" \
        --port "$PORT" --key-file "$KEY" --log "$ROOT/logs/worker-$SLURM_JOB_ID.log" || true
    echo "$(date -Is) server exited; restarting in 10 s"
    sleep 10
done
RUN
