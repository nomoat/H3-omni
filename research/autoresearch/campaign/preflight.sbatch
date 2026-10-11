#!/bin/bash
#SBATCH --job-name=h3-search-preflight
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-100/preflight-%j.log
set -euo pipefail
H3_SEARCH=/home/cj/h3-cuda/autoresearch-100
H3_STAGE=/var/tmp/h3-search-100
mkdir -p "$H3_STAGE/seed" "$H3_STAGE/harness" "$H3_STAGE/preflight" "$H3_SEARCH/preflight"
tar -xf /home/cj/h3-cuda/releases/14274/source.tar -C "$H3_STAGE/seed"
cp "$H3_SEARCH/harness/runtime.py" "$H3_STAGE/harness/"
cp "$H3_SEARCH/prompt.txt" "$H3_STAGE/prompt.txt"
trap 'cp -a "$H3_STAGE/preflight/." "$H3_SEARCH/preflight/"' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/var/tmp/h3-cuda-14274/python
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/var/tmp/h3-cuda-14274/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
nvidia-smi > preflight/device.txt
df -h /var/tmp > preflight/disk.txt
python3 harness/runtime.py build seed /var/tmp/h3-cuda-14274 preflight/build
python3 harness/runtime.py benchmark seed /var/tmp/h3-cuda-14228/local-model/FL2VA prompt.txt preflight/baseline --samples 3
RUN
