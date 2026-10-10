#!/bin/bash
#SBATCH --job-name=h3-serve-check
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cj/h3-serve/serve-%j.log
# Serving check plus acceptance, in the default (fast) mode and in the
# reference mode (H3_FLASH3=0 H3_VAE_3XTF32=0: FlashAttention-2 and the FP32
# video VAE, which match PyTorch bit for bit).
# 1. Default mode: one process serves seven requests (new prompts, repeated
#    prompts, other canvases, durations and step counts) with resident
#    weights; every distinct request is rerun as a fresh one-shot process;
#    latents, decoded RGB, PCM and MP4 must match bit for bit.
# 2. Reference mode: the one-shot MP4 equals the accepted release 14274, and
#    the original PyTorch acceptance passes unchanged.
# 3. Default mode: decoder acceptance on the same latents, and the calibrated
#    trajectory acceptance (tests/pytorch/calibrated_accept.py).
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/cj/h3-serve
STAGE=/var/tmp/h3-serve-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
SOURCE=${SOURCE:-$ROOT/source.tar}
mkdir -p "$STAGE/icu" "$STAGE/python" "$STAGE/diffusers" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
mkdir -p "$STAGE/.deps"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
tar -xf /home/cj/h3-cuda/python-deps.tar -C "$STAGE/python"
tar -xf /home/cj/h3-cuda/diffusers.tar -C "$STAGE/diffusers"
cp "$SOURCE" "$RESULT/source.tar"
SHARED_MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
LOCAL_MODEL=/var/tmp/h3-cuda-14228/local-model
MODEL=$SHARED_MODEL
if [ -f "$LOCAL_MODEL/FL2VA/transformer/config.json" ]; then MODEL=$LOCAL_MODEL; fi
echo "serving model: $MODEL"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$SHARED_MODEL" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"
SHARED_MODEL="$3"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
mkdir -p results/serve results/fresh results/release results/accept results/default
echo "$MODEL" > results/model-path.txt
nvidia-smi > results/device.txt
nvcc --version > results/cuda-version.txt
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" all
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check"
sha256sum build-cuda/h3 build-cuda/h3_serve_check build-cuda/*.so > results/binary-sha256.txt

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
cat results/serve/events.jsonl

# Fresh one-shot references with the same new binary (cache disabled path).
while IFS='|' read -r name width height frames steps seed prompt; do
    case "$name" in r3-*|r5-*|r7-*) continue;; esac
    mkdir -p "results/fresh/$name"
    start=$(date +%s.%N)
    H3_DUMP_DIR="results/fresh/$name" build-cuda/h3 -d "$MODEL" -p "$prompt" \
        --width "$width" --height "$height" --frames "$frames" --steps "$steps" \
        --layers 50 --reuse 1 --core-reuse 1 --seed "$seed" \
        -o "results/fresh/$name/output.mp4" 2> "results/fresh/$name/stderr.log"
    end=$(date +%s.%N)
    echo "{\"event\":\"fresh\",\"name\":\"$name\",\"seconds\":$(python3 -c "print($end-$start)")}" \
        | tee -a results/fresh/events.jsonl
done < results/requests.txt

# The accepted release binary for one request: the reference-mode one-shot
# path is unchanged.
H3_FLASH3=0 H3_VAE_3XTF32=0 build-cuda/h3 -d "$MODEL" -p "$BOAT" \
    --width 864 --height 480 --frames 56 --steps 20 --layers 50 --reuse 1 \
    --core-reuse 1 --seed 7 -o results/release/r1-boat-reference.mp4 2> results/release/reference.log
start=$(date +%s.%N)
LD_LIBRARY_PATH="/home/cj/h3-cuda/releases/14274/lib:$LD_LIBRARY_PATH" \
    /home/cj/h3-cuda/releases/14274/h3 -d "$MODEL" -p "$BOAT" \
    --width 864 --height 480 --frames 56 --steps 20 --layers 50 --reuse 1 \
    --core-reuse 1 --seed 7 -o results/release/r1-boat.mp4 2> results/release/stderr.log
end=$(date +%s.%N)
echo "{\"event\":\"release\",\"name\":\"r1-boat\",\"seconds\":$(python3 -c "print($end-$start)")}" \
    | tee results/release/events.jsonl
sha256sum results/release/r1-boat.mp4 results/release/r1-boat-reference.mp4 \
    | tee results/release/mp4-sha256.txt
[ "$(cut -d' ' -f1 results/release/mp4-sha256.txt | uniq | wc -l)" = 1 ]

python3 tests/serving/compare.py --serve results/serve --fresh results/fresh \
    --pairs r1-boat=r1-boat,r2-fox=r2-fox,r3-boat=r1-boat,r4-cat=r4-cat,r5-fox=r2-fox,r6-city=r6-city,r7-fox=r2-fox \
    --output results/serve/compare.json

# Unchanged phase-one acceptance against the new build in reference mode.
export H3_FLASH3=0 H3_VAE_3XTF32=0
H3_WEIGHTS="$SHARED_MODEL/FL2VA"
A=results/accept
cp tests/pytorch/sources.json tests/pytorch/acceptance.json "$A/"
cp results/cuda-version.txt results/device.txt results/binary-sha256.txt "$A/"
python3 tests/pytorch/primitives.py --output "$A/primitives.json"
python3 tests/pytorch/host.py --output "$A/host.json"
python3 tests/pytorch/text.py --weights "$H3_WEIGHTS" --output "$A/text"
python3 tests/pytorch/trajectory_suite.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --text "$A/text/text-0.safetensors" --output "$A/trajectories"
python3 tests/pytorch/decoder_suite.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --trajectories "$A/trajectories" --output "$A/decoded"
python3 tests/pytorch/cli_smoke.py --weights "$H3_WEIGHTS" --output "$A"
python3 tests/pytorch/accept.py "$A"
python3 tests/pytorch/extended.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --output "$A/extended"
unset H3_FLASH3 H3_VAE_3XTF32
# Default mode: decoder acceptance on the reference-mode latents, then the
# calibrated trajectory acceptance.
python3 tests/pytorch/decoder_suite.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --trajectories "$A/trajectories" --output results/default/decoded
python3 tests/pytorch/calibrate.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --text "$A/text/text-0.safetensors" --output results/default/trajectories | grep -v '^{"label"'
python3 tests/pytorch/calibrated_accept.py --candidate results/default/trajectories/summary.json \
    --output results/default/calibrated-acceptance.json
RUN
