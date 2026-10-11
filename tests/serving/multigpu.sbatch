#!/bin/bash
#SBATCH --job-name=h3-multigpu
#SBATCH --partition=h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=02:30:00
#SBATCH --output=/home/cj/h3-serve/multigpu-%j.log
# Multi-GPU DiT (H3_GPUS) check. Submit with --gres=gpu:N.
#  1. correctness: two short requests with H3_GPUS=1 and H3_GPUS=N, dumps
#     compared by tests/serving/compare_gpus.py;
#  2. latency: warm 5 s and 10 s requests with H3_GPUS=N (no dumps).
# Usage: sbatch --gres=gpu:N multigpu.sbatch SOURCE_TAR N [CHECK=1]
set -euo pipefail
ROOT=/home/cj/h3-serve
SOURCE=$1
N=$2
CHECK=${3:-1}
STAGE=/var/tmp/h3-multigpu-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$STAGE/python" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
tar -xf /home/cj/h3-cuda/python-deps.tar -C "$STAGE/python"
SHARED_MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
LOCAL_MODEL=/var/tmp/h3-cuda-14228/local-model
MODEL=$SHARED_MODEL
if [ -f "$LOCAL_MODEL/FL2VA/transformer/config.json" ]; then MODEL=$LOCAL_MODEL; fi
echo "node $(hostname) N=$N model=$MODEL"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$N" "$CHECK" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"; N="$3"; CHECK="$4"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
mkdir -p results
nvidia-smi topo -m > results/topology.txt || true
make -f cuda/Makefile -j16 ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check"
P1="A small wooden fishing boat moves slowly across a calm mountain lake at sunrise. Mist drifts between pine trees, gentle waves touch the rocks, and birds call in the distance."
P2="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
P3="A street musician plays an acoustic guitar on a busy city corner at dusk. People walk past, car lights blur in the background, and the music mixes with traffic noise."
if [ "$CHECK" = 1 ]; then
    printf '%s\n' "c1-boat|864|480|56|20|7|$P1" "c2-cat|512|512|22|20|3|$P2" > results/check.txt
    for g in 1 "$N"; do
        mkdir -p "results/check-g$g"
        H3_GPUS=$g build-cuda/h3_serve_check "$MODEL" "results/check-g$g" results/check.txt \
            > "results/check-g$g/events.jsonl" 2> "results/check-g$g/stderr.log" || echo "check g$g exit $?"
        tail -3 "results/check-g$g/stderr.log"
        cat "results/check-g$g/events.jsonl"
    done
    python3 tests/serving/compare_gpus.py --one results/check-g1 --many "results/check-g$N" \
        --output results/compare.json
fi
printf '%s\n' "warmup|864|480|56|20|1|$P2" \
    "s5-boat|864|480|120|20|7|$P1" "s5-fox|864|480|120|20|11|$P2" \
    "s10-boat|864|480|240|20|7|$P1" "s10-fox|864|480|240|20|11|$P2" \
    "s10-guitar|864|480|240|20|23|$P3" > results/requests.txt
mkdir -p results/serve
nvidia-smi --query-gpu=index,timestamp,memory.used,clocks.sm,power.draw --format=csv,noheader -lms 1000 \
    > results/serve/gpu-samples.csv &
SAMPLER=$!
H3_GPUS=$N H3_SERVE_DUMP=0 build-cuda/h3_serve_check "$MODEL" results/serve results/requests.txt \
    > results/serve/events.jsonl 2> results/serve/stderr.log || echo "serve exit $?"
kill $SAMPLER || true
tail -3 results/serve/stderr.log
cat results/serve/events.jsonl
RUN
