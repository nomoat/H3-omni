#!/bin/bash
#SBATCH --job-name=h3-search-100-fast
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-100/search-fast-%j.log
set -euo pipefail
H3_STAGE=/var/tmp/h3-search-100
H3_SHARED=/home/cj/h3-cuda/autoresearch-100
mkdir -p "$H3_STAGE/control-fast" "$H3_SHARED/results-fast"
install -m 0644 "$H3_SHARED/worker_fast.py" "$H3_STAGE/control-fast/worker_fast.py"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" "$H3_SHARED" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/var/tmp/h3-cuda-14274/diffusers/src:/var/tmp/h3-cuda-14274/python
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/var/tmp/h3-cuda-14274/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
python3 "$1/control-fast/worker_fast.py" --stage "$1" --shared "$2/results-fast" \
  --oracle "$1/oracle" --oracle-sha256 f38bc6b6d80fec1b5dbcbe0233293ddb294e3403cdcc9b0ad6c303af8235764c \
  --baseline-json "$2/fast-inputs/baseline-two.json" --legacy-manifest "$2/fast-inputs/legacy.json" --limit 100
RUN
