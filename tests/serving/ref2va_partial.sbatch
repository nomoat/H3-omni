#!/bin/bash
#SBATCH --job-name=h3-ref2va-partial
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-serve/ref2va-partial-%j.log
# A model directory with only Ref2VA/transformer: prompt-only requests must
# still run, and an ordered-reference request must name the missing file.
# Usage: sbatch ref2va_partial.sbatch SOURCE_TAR
set -euo pipefail
STAGE=/var/tmp/h3-ref2va-partial-$SLURM_JOB_ID
mkdir -p "$STAGE/src/.deps" "$STAGE/icu" "$STAGE/model/Ref2VA"
tar -xf "$1" -C "$STAGE/src"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/src/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/src/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
REAL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
ln -s "$REAL/model_index.json" "$STAGE/model/model_index.json"
ln -s "$REAL/FL2VA" "$STAGE/model/FL2VA"
ln -s "$REAL/Ref2VA/transformer" "$STAGE/model/Ref2VA/transformer"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" <<'RUN'
set -uo pipefail
cd "$1/src"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export H3_FFPROBE=/home/cj/h3-serve/tools/ffmpeg-7.0.2-static/ffmpeg-7.0.2-amd64-static/ffprobe
make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr" "$1/src/build-cuda/h3" > build.log 2>&1 || { tail -20 build.log; exit 1; }
build-cuda/h3 -d "$1/model" -p "A guitarist plays on a street corner." --ref-image /home/cj/h3-serve/inputs/guitar-ref.png \
    --width 256 --height 256 --frames 22 --steps 4 -o "$1/ref.mp4" 2> ref.err
echo "ordered reference exit $? : $(grep '^h3:' ref.err)"
build-cuda/h3 -d "$1/model" -p "A cat walks through a sunny garden." \
    --width 256 --height 256 --frames 22 --steps 4 -o "$1/prompt.mp4" 2> prompt.err
echo "prompt-only exit $? : $(ls -la $1/prompt.mp4 2>&1 | awk '{print $5, $9}')"
RUN
