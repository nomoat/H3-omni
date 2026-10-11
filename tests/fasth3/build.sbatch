#!/bin/bash
#SBATCH --job-name=h3-fasth3-build
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-h200x1-2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-fasth3/build-%j.log
set -euo pipefail
ROOT=/home/cj/h3-fasth3
STAGE=/var/tmp/h3-fasth3
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE" "$RESULT"
tar -xmf "$ROOT/source.tar" -C "$STAGE"
if [ ! -d "$STAGE/.deps/flash-cuda" ]; then
    mkdir -p "$STAGE/.deps" "$STAGE/icu" "$STAGE/python" "$STAGE/diffusers"
    tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
    tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
    tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
    tar -xf /home/cj/h3-cuda/python-deps.tar -C "$STAGE/python"
    tar -xf /home/cj/h3-cuda/diffusers.tar -C "$STAGE/diffusers"
fi
cp "$ROOT/source.tar" "$RESULT/source.tar"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi > "$2/device.txt"
make -f cuda/Makefile -j8 VSA=1 ICU_ROOT="$1/icu/usr" all
python3 tests/fasth3/vsa.py --fastvideo reference --output "$2/vsa.json"
python3 tests/fasth3/contract.py --output "$2/contract.json"
python3 tests/pytorch/host.py --output "$2/host.json"
RUN
