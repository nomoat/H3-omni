#!/bin/bash
#SBATCH --job-name=h3-fasth3-regression
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-h200x1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-fasth3/regression-%j.log
set -euo pipefail
STAGE=/var/tmp/h3-fasth3
RESULT=/home/cj/h3-fasth3/results/$SLURM_JOB_ID
mkdir -p "$RESULT"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FLASH3=0 H3_VAE_3XTF32=0
python3 tests/pytorch/primitives.py --output "$2/primitives.json"
python3 tests/pytorch/real_dit.py \
    --weights /data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA \
    --diffusers "$1/diffusers" --text /home/cj/h3-cuda/results/14274/text/text-0.safetensors \
    --width 256 --height 256 --frames 22 --steps 20 --seed 0 --output "$2/original"
RUN
