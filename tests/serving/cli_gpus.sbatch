#!/bin/bash
#SBATCH --job-name=h3-cli-gpus
#SBATCH --partition=h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=00:40:00
#SBATCH --output=/home/cj/h3-serve/cli-gpus-%j.log
# CLI smoke test of --gpus: one 512x512 request on 2 GPUs, and --gpus 3,
# which must be refused (3 does not divide the 56 heads).
# Usage: sbatch cli_gpus.sbatch SOURCE_TAR
set -euo pipefail
STAGE=/var/tmp/h3-cli-gpus-$SLURM_JOB_ID
RESULT=/home/cj/h3-serve/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$RESULT"
tar -xf "$1" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
if [ -f /var/tmp/h3-cuda-14228/local-model/FL2VA/transformer/config.json ]; then
    MODEL=/var/tmp/h3-cuda-14228/local-model; fi
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
make -f cuda/Makefile -j16 ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3" > /dev/null 2>&1
build-cuda/h3 --help 2>&1 | grep -A1 -- --gpus || true
build-cuda/h3 -d "$2" -p "A cat walks through a sunny garden." --width 512 --height 512 \
    --frames 22 --seed 0 --gpus 2 -o "$3/cat-2gpus.mp4" 2>&1 | tail -4
ls -la "$3/cat-2gpus.mp4"
build-cuda/h3 -d "$2" -p "A cat." --width 512 --height 512 --frames 22 --gpus 3 \
    -o "$3/never.mp4" 2>&1 | tail -2 || true
RUN
