#!/bin/bash
#SBATCH --job-name=h3-elementwise
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-serve/elementwise-%j.log
# Bit-exactness and timing of the hand-written DiT elementwise kernels
# (cuda/fast_primitives.cuh) against the generated kernels, then a build of the
# CUDA library to catch compile errors.
# Usage: sbatch elementwise.sbatch SOURCE_TAR
set -euo pipefail
SOURCE=$1
STAGE=/var/tmp/h3-elementwise-$SLURM_JOB_ID
mkdir -p "$STAGE"
tar -xf "$SOURCE" -C "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
apptainer exec --nv --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" <<'RUN'
set -euo pipefail
cd "$1"
nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv
nvcc -O2 -std=c++17 -arch=sm_90 --fmad=false -I. -Icuda \
    tests/serving/profile/elementwise_bench.cu -o elementwise_bench
./elementwise_bench
nvcc -O2 -lineinfo -std=c++17 -arch=sm_90 --fmad=false -Xcompiler=-fPIC -I. -Icuda \
    -I/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/include \
    -c cuda/h3_gpu_cuda.cu -o h3_gpu_cuda.o --resource-usage > resource.txt 2>&1 || { cat resource.txt; exit 1; }
grep -A1 "h3_fast" resource.txt || true
echo build ok
RUN
