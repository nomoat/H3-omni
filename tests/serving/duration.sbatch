#!/bin/bash
#SBATCH --job-name=h3-duration
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cj/h3-serve/duration-%j.log
# Resident-serving latency for 5 s and 10 s videos at 864x480, 20 steps.
# One process: a 56-frame warm-up (absorbs weight loading), then three
# 5 s requests and three 10 s requests with different prompts and seeds.
set -euo pipefail
ROOT=/home/cj/h3-serve
STAGE=/var/tmp/h3-duration-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
SOURCE=${1:-${SOURCE:-$ROOT/source.tar}}
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
SHARED_MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
LOCAL_MODEL=/var/tmp/h3-cuda-14228/local-model
MODEL=$SHARED_MODEL
if [ -f "$LOCAL_MODEL/FL2VA/transformer/config.json" ]; then MODEL=$LOCAL_MODEL; fi
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export H3_SERVE_DUMP=0
mkdir -p results/serve
nvidia-smi > results/device.txt
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check"
P1="A small wooden fishing boat moves slowly across a calm mountain lake at sunrise. Mist drifts between pine trees, gentle waves touch the rocks, and birds call in the distance."
P2="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
P3="A street musician plays an acoustic guitar on a busy city corner at dusk. People walk past, car lights blur in the background, and the music mixes with traffic noise."
cat > results/requests.txt <<EOF
warmup|864|480|56|20|1|$P2
s5-boat|864|480|120|20|7|$P1
s5-fox|864|480|120|20|11|$P2
s5-guitar|864|480|120|20|23|$P3
s10-boat|864|480|240|20|7|$P1
s10-fox|864|480|240|20|11|$P2
s10-guitar|864|480|240|20|23|$P3
EOF
nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -lms 500 > results/serve/gpu-memory.csv &
SAMPLER=$!
build-cuda/h3_serve_check "$MODEL" results/serve results/requests.txt \
    > results/serve/events.jsonl 2> results/serve/stderr.log || echo "serve exit $?"
kill "$SAMPLER" || true
cat results/serve/events.jsonl
for f in results/serve/*/output.mp4; do
    echo "$f $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null || true)"
done
RUN
