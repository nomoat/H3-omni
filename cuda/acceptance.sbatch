#!/bin/bash
#SBATCH --job-name=h3-cuda-accept
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cj/h3-cuda/acceptance-%j.log
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
cp "$H3_ROOT/source.tar" "$H3_RESULT/source.tar"
cp "$H3_ROOT/reference-requirements.lock" "$H3_RESULT/"
trap 'cp -a "$H3_STAGE/results/." "$H3_RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
H3_WEIGHTS="${H3_WEIGHTS:-/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA}"
mkdir -p results
cp tests/pytorch/sources.json tests/pytorch/acceptance.json results/
nvcc --version > results/cuda-version.txt
nvidia-smi > results/device.txt
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr"
sha256sum build-cuda/h3 build-cuda/*.so > results/binary-sha256.txt
python3 tests/pytorch/primitives.py --output results/primitives.json
python3 tests/pytorch/host.py --output results/host.json
python3 tests/pytorch/text.py --weights "$H3_WEIGHTS" --output results/text
python3 tests/pytorch/trajectory_suite.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --text results/text/text-0.safetensors --output results/trajectories
python3 tests/pytorch/decoder_suite.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --trajectories results/trajectories --output results/decoded
python3 tests/pytorch/cli_smoke.py --weights "$H3_WEIGHTS" --output results
python3 tests/pytorch/accept.py results
python3 tests/pytorch/extended.py --weights "$H3_WEIGHTS" --diffusers "$1/diffusers" \
    --output results/extended
RUN
