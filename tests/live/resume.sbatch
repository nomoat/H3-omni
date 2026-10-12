#!/bin/bash
#SBATCH --job-name=h3-live
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cj/h3-live/resume-%j.log
set -euo pipefail
LIVE_STAGE=$1
LIVE_SOURCE=$2
LIVE_RESULT=/home/cj/h3-live/results/$SLURM_JOB_ID
mkdir -p "$LIVE_RESULT"
tar -xf "$LIVE_SOURCE" -C "$LIVE_STAGE"
apptainer exec --nv --bind /data0:/data0,/var/tmp:/var/tmp,/home/cj:/home/cj \
 /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$LIVE_STAGE" "$LIVE_RESULT" "${3:-1}" "${4:-256}" "${5:-256}" <<'RUN'
set -euo pipefail
cd "$1"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr"
python3 tests/live/attention.py --output "$2/attention.json"
python3 tests/live/lora.py --output "$2/lora.json"
export H3_DUMP_DIR="$2/latents"
mkdir -p "$H3_DUMP_DIR"
build-cuda/h3 -d /data0/models/h3-taomate/native --mode live \
 -p 'A woman with brown hair sits at a desk in a bright studio. She looks at the camera, smiles and says, "Welcome to the show. Today we will explore something new together." A steady medium shot.' \
 --width "$4" --height "$5" --frames 243 --seed 42 --gpus "$3" \
 -o "$2/live-10s.mp4" > "$2/generation.log" 2>&1
cp build-cuda/h3 "$2/h3"
tail -c 3000 "$2/generation.log"
RUN
