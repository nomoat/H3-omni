#!/bin/bash
#SBATCH --job-name=h3-fasth3-merged
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-h200x1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cj/h3-fasth3/merged-%j.log
# FastH3 (VSA=1 build) from a given source tar: contract and four-step parity
# against the pinned reference (512x512, 22 frames, as in default.sbatch),
# decoders, then resident serving latency for 5 s and 10 s videos.
# Reuses the dependencies staged in /var/tmp/h3-fasth3 on this node.
# Usage: sbatch merged.sbatch SOURCE_TAR
set -euo pipefail
BASE=/var/tmp/h3-fasth3
STAGE=/var/tmp/h3-fasth3-merged-$SLURM_JOB_ID
RESULT=/home/cj/h3-fasth3/results/$SLURM_JOB_ID
mkdir -p "$STAGE" "$RESULT"
tar -xf "$1" -C "$STAGE"
for d in .deps icu python diffusers reference; do ln -s "$BASE/$d" "$STAGE/$d"; done
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
nvidia-smi > "$2/device.txt"
make -f cuda/Makefile -j8 VSA=1 ICU_ROOT="$1/icu/usr" all "$1/build-cuda/h3_serve_check"
MODEL=/data0/models/h3-fasth3/vsa-native
python3 tests/fasth3/contract.py --output "$2/contract.json"
python3 tests/fasth3/parity.py --source /data0/models/h3-fasth3/vsa-source \
    --fastvideo reference --oracle hopper --weights $MODEL/FL2VA \
    --diffusers "$1/diffusers" --text /home/cj/h3-cuda/results/14274/text/text-1.safetensors \
    --width 512 --height 512 --frames 22 --seed 1 --output "$2/parity" 2>&1 | grep -v Warning
python3 tests/pytorch/decoders.py --weights $MODEL/FL2VA \
    --diffusers "$1/diffusers" --latents "$2/parity/latents.safetensors" --output "$2/decoded" \
    2>&1 | grep -v Warning
P1="A small wooden fishing boat moves slowly across a calm mountain lake at sunrise. Mist drifts between pine trees, gentle waves touch the rocks, and birds call in the distance."
P2="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
mkdir -p "$2/serve"
cat > "$2/requests.txt" <<REQ
warmup|864|480|56|4|1|$P2
s5-boat|864|480|120|4|7|$P1
s5-fox|864|480|120|4|11|$P2
s10-boat|864|480|240|4|7|$P1
s10-fox|864|480|240|4|11|$P2
REQ
H3_SERVE_DUMP=0 build-cuda/h3_serve_check $MODEL "$2/serve" "$2/requests.txt" \
    > "$2/serve/events.jsonl" 2> "$2/serve/stderr.log" || echo "serve exit $?"
cat "$2/serve/events.jsonl"
RUN
