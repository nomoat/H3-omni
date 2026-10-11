#!/bin/bash
#SBATCH --job-name=h3-fasth3-smoke
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-h200x1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-fasth3/smoke-%j.log
set -euo pipefail
STAGE=/var/tmp/h3-fasth3
RESULT=/home/cj/h3-fasth3/results/$SLURM_JOB_ID
mkdir -p "$RESULT"
tar -xmf /home/cj/h3-fasth3/source.tar -C "$STAGE"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export H3_FFPROBE=/home/cj/h3-serve/tools/ffmpeg-7.0.2-static/ffmpeg-7.0.2-amd64-static/ffprobe
make -f cuda/Makefile -j8 VSA=1 ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check"
ldd build-cuda/h3 > "$2/ldd.txt"
python3 tests/fasth3/smoke.py --weights /data0/models/h3-fasth3/vsa-native --output "$2/smoke"
cp build-cuda/h3 "$2/h3"
sha256sum build-cuda/h3 build-cuda/libh3_model.so > "$2/binaries.sha256"
RUN
