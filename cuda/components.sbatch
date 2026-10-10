#!/bin/bash
#SBATCH --job-name=h3-cuda-text
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cj/h3-cuda/components-%j.log
set -euo pipefail
export PYTHONNOUSERSITE=1
H3_ROOT=/home/cj/h3-cuda
H3_STAGE=/var/tmp/h3-cuda-$SLURM_JOB_ID
H3_RESULT="$H3_ROOT/results/$SLURM_JOB_ID"
mkdir -p "$H3_STAGE/icu" "$H3_STAGE/python" "$H3_STAGE/diffusers" "$H3_RESULT"
tar -xf "$H3_ROOT/source.tar" -C "$H3_STAGE"
mkdir -p "$H3_STAGE/.deps"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$H3_STAGE/.deps"
tar -xf "$H3_ROOT/icu.tar" -C "$H3_STAGE/icu"
tar -xf "$H3_ROOT/python-deps.tar" -C "$H3_STAGE/python"
tar -xf "$H3_ROOT/diffusers.tar" -C "$H3_STAGE/diffusers"
trap 'cp -a "$H3_STAGE/results/." "$H3_RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS=8
mkdir -p results
make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr"
python3 tests/pytorch/primitives.py --output results/primitives.json
python3 tests/pytorch/text.py \
    --weights "${H3_TEXT_WEIGHTS:-/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA}" \
    --output results/text
RUN
