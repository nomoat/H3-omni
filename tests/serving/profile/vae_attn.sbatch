#!/bin/bash
#SBATCH --job-name=h3-vaeattn
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-serve/vaeattn-%j.log
# Video-VAE attention: current FP32 path vs fused 3xTF32 kernel, speed and
# error against FP64. Usage: sbatch vae_attn.sbatch SOURCE_TAR
set -euo pipefail
STAGE=/var/tmp/h3-vaeattn-$SLURM_JOB_ID
mkdir -p "$STAGE"
tar -xf "$1" -C "$STAGE"
apptainer exec --nv --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" <<'RUN'
set -euo pipefail
cd "$1"
# Same floating-point flags as the backend (cuda/Makefile NVFLAGS).
nvcc -O2 -std=c++17 -arch=sm_90 --fmad=false -Xptxas -v -Icuda tests/serving/profile/vae_attn_bench.cu \
    -o vae_attn_bench -lcublas > build.log 2>&1 || { cat build.log; exit 1; }
grep -E "attention|registers|spill" build.log || true
# The backend translation unit must also compile with the kernel.
nvcc -O2 -std=c++17 -arch=sm_90 --fmad=false -I. -Icuda \
    -I/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/include \
    -c cuda/h3_gpu_cuda.cu -o h3_gpu_cuda.o && echo "backend compiles"
./vae_attn_bench
./vae_attn_bench 1001 32
RUN
rm -rf "$STAGE"
