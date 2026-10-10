#!/bin/bash
#SBATCH --job-name=h3-cuda-build
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cj/h3-cuda/build-%j.log
set -euo pipefail
export PYTHONNOUSERSITE=1
H3_STAGE=/var/tmp/h3-cuda-$SLURM_JOB_ID
mkdir -p "$H3_STAGE"
tar -xf /home/cj/h3-cuda/source.tar -C "$H3_STAGE"
mkdir -p "$H3_STAGE/.deps"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$H3_STAGE/.deps"
mkdir -p "$H3_STAGE/icu"
tar -xf /home/cj/h3-cuda/icu.tar -C "$H3_STAGE/icu"
trap 'mkdir -p /home/cj/h3-cuda/results; cp "$H3_STAGE"/*results.json /home/cj/h3-cuda/results/ 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind "$H3_STAGE:$H3_STAGE" \
  /data0/apptainer/pytorch_291_cu128.sif \
  bash -c 'set -e; cd "$1"; make -f cuda/Makefile -j8 ICU_ROOT="$1/icu/usr"; ./build-cuda/h3 --help; python3 tests/pytorch/primitives.py --output primitive-results.json' bash "$H3_STAGE"
mkdir -p /home/cj/h3-cuda/build
cp "$H3_STAGE"/build-cuda/h3 "$H3_STAGE"/build-cuda/*.so /home/cj/h3-cuda/build/
