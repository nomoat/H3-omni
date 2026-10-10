#!/bin/bash
#SBATCH --job-name=h3-vaegemm
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-serve/vaegemm-%j.log
# Video-VAE GEMM precision/speed options. Usage: sbatch vae_gemm.sbatch SOURCE_TAR
set -euo pipefail
STAGE=/var/tmp/h3-vaegemm-$SLURM_JOB_ID
mkdir -p "$STAGE/.deps"
tar -xf "$1" -C "$STAGE"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
apptainer exec --nv --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" <<'RUN'
set -euo pipefail
cd "$1"
nvcc -O3 -std=c++17 -arch=sm_90 -I.deps/cutlass-flash3/include tests/serving/profile/vae_gemm_bench.cu \
    -o vae_gemm_bench -lcublas
./vae_gemm_bench
RUN
