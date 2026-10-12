#!/bin/bash
#SBATCH --job-name=h3-live-regression
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cj/h3-live/regression-%j.log
set -euo pipefail
LIVE_STAGE=$1
if [ -n "${2:-}" ]; then tar -xf "$2" -C "$LIVE_STAGE"; fi
LIVE_RESULT=/home/cj/h3-live/results/$SLURM_JOB_ID
mkdir -p "$LIVE_RESULT"
tar -xzf /home/cj/h3-fasth3/tk-deps.tar.gz -C "$LIVE_STAGE/.deps"
apptainer exec --nv --bind /data0:/data0,/var/tmp:/var/tmp,/home/cj:/home/cj \
 /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$LIVE_STAGE" "$LIVE_RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
make -f cuda/Makefile -j8 VSA=1 ICU_ROOT="$1/icu/usr"
python3 tests/pytorch/primitives.py --output "$2/primitives.json"
python3 tests/live/attention.py --output "$2/attention.json"
python3 tests/live/lora.py --output "$2/lora.json"
python3 tests/live/sharding.py --output "$2/sharding.json"
build-cuda/h3 --mode original --info -d /data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe > "$2/original-info.txt" 2>&1
build-cuda/h3 --mode fast --info -d /data0/models/h3-fasth3/vsa-native > "$2/fast-info.txt" 2>&1
build-cuda/h3 --mode live --info -d /data0/models/h3-taomate/native > "$2/live-info.txt" 2>&1
if build-cuda/h3 --mode live --info -d /data0/models/h3-fasth3/vsa-native > "$2/mismatch.txt" 2>&1; then
 echo 'mode mismatch was incorrectly accepted'; exit 1
fi
RUN
