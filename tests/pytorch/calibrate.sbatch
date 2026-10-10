#!/bin/bash
#SBATCH --job-name=h3-calibrate
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cj/h3-serve/calibrate-%j.log
# Trajectory-sensitivity calibration over the required case matrix.
# Usage: sbatch calibrate.sbatch SOURCE_TAR MODE
#   MODE=math : native FlashAttention-2 (bit-exact PyTorch default) vs PyTorch math SDPA
#   MODE=fa3  : native FlashAttention-3 vs PyTorch default SDPA
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/cj/h3-serve
SOURCE=$1
MODE=$2
STAGE=/var/tmp/h3-calibrate-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/python" "$STAGE/diffusers" "$STAGE/.deps" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
tar -xf /home/cj/h3-cuda/python-deps.tar -C "$STAGE/python"
tar -xf /home/cj/h3-cuda/diffusers.tar -C "$STAGE/diffusers"
WEIGHTS=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA
TEXT=$ROOT/results/14416/accept/text/text-0.safetensors
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true; find "$RESULT" -name "*.safetensors" -size +50M -delete 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$WEIGHTS" "$TEXT" "$MODE" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/libh3_model.so"
EXTRA=()
if [ "$4" = math ]; then export H3_FLASH3=0; EXTRA=(--attention-math-diagnostic); fi
mkdir -p results
python3 tests/pytorch/calibrate.py --weights "$2" --diffusers "$1/diffusers" --text "$3" \
    --output results/$4 "${EXTRA[@]}" 2>&1 | grep -v '^{"label"'
cat results/$4/summary.json
RUN
