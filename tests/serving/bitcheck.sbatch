#!/bin/bash
#SBATCH --job-name=h3-bitcheck
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cj/h3-serve/bitcheck-%j.log
# Bit-exactness gate for changes that must not alter values (allocation,
# scheduling, launch shape, fusion with the same operation order). Serves the
# job-14416 request sequence and compares every dump with archived fresh
# one-shot outputs. The default reference is job 14416, which used the
# reference mode (FlashAttention-2, FP32 video VAE); with that reference the
# run sets H3_FLASH3=0 H3_VAE_3XTF32=0. Pass MODE=default together with a
# fresh directory produced by the default mode to check the fast path.
# Usage: sbatch bitcheck.sbatch SOURCE_TAR [REFERENCE_FRESH_DIR [MODE]]
set -euo pipefail
ROOT=/home/cj/h3-serve
SOURCE=$1
REFERENCE=${2:-$ROOT/results/14416/fresh}
MODE=${3:-reference}
if [ "$MODE" = reference ]; then export H3_FLASH3=0 H3_VAE_3XTF32=0; fi
STAGE=/var/tmp/h3-bitcheck-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
LOCAL_MODEL=/var/tmp/h3-cuda-14228/local-model
if [ -f "$LOCAL_MODEL/FL2VA/transformer/config.json" ]; then MODEL=$LOCAL_MODEL; fi
echo "node $(hostname) model $MODEL reference $REFERENCE mode $MODE"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true; rm -f "$RESULT"/serve/*/*.f32' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$REFERENCE" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"
REFERENCE="$3"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
mkdir -p results/serve
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check"
BOAT=$(python3 -c 'import json;print(json.load(open("tests/pytorch/extended.json"))["prompt"])')
FOX="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
CAT="A cat walks through a sunny garden."
CITY="一只橘色的小猫在下雨的城市街道上奔跑，霓虹灯倒映在水坑里，远处传来汽车的声音。"
cat > results/requests.txt <<EOF
r1-boat|864|480|56|20|7|$BOAT
r2-fox|864|480|56|20|11|$FOX
r3-boat|864|480|56|20|7|$BOAT
r4-cat|512|512|22|20|0|$CAT
r5-fox|864|480|56|20|11|$FOX
r6-city|256|256|22|12|42|$CITY
r7-fox|864|480|56|20|11|$FOX
EOF
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 \
    > results/serve/gpu-memory.csv &
SAMPLER=$!
build-cuda/h3_serve_check "$MODEL" results/serve results/requests.txt \
    > results/serve/events.jsonl 2> results/serve/stderr.log
kill "$SAMPLER" || true
python3 -c '
import json,sys
for l in open(sys.argv[1]):
    e=json.loads(l)
    if e["event"]=="request":
        p={x["phase"]:x for x in e["phases"]}
        d=p["denoise"]
        print(e["name"], round(e["seconds"],2), "denoise", round(d["last"]-d["first"],2))
' results/serve/events.jsonl
awk -F, '{print $2}' results/serve/gpu-memory.csv | sort -n | tail -1
python3 tests/serving/compare.py --serve results/serve --fresh "$REFERENCE" \
    --pairs r1-boat=r1-boat,r2-fox=r2-fox,r3-boat=r1-boat,r4-cat=r4-cat,r5-fox=r2-fox,r6-city=r6-city,r7-fox=r2-fox \
    --output results/serve/compare.json
RUN
