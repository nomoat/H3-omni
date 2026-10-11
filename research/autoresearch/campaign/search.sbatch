#!/bin/bash
#SBATCH --job-name=h3-autoresearch-100
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-100/search-%j.log
set -euo pipefail
H3_STAGE=/var/tmp/h3-search-100
H3_SHARED=/home/cj/h3-cuda/autoresearch-100
H3_JUDGE="$H3_STAGE/seed/research/autoresearch/campaign"
mkdir -p "$H3_SHARED/results"
tar -xf "$H3_SHARED/harness.tar" -C "$H3_JUDGE"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" "$H3_SHARED" <<'RUN'
set -euo pipefail
cd "$1"
H3_SHARED="$2"
H3_JUDGE="$1/seed/research/autoresearch/campaign"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/var/tmp/h3-cuda-14274/diffusers/src:/var/tmp/h3-cuda-14274/python
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/var/tmp/h3-cuda-14274/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
H3_ORACLE_HASH=$(python3 -c 'import json;print(json.load(open("oracle/verification.json"))["manifest_sha256"])')
nvidia-smi > "$H3_SHARED/search-device-$SLURM_JOB_ID.txt"
if [ ! -f "$H3_SHARED/oracle.tar" ]; then
  tar -cf "$H3_SHARED/oracle.tar.pending" oracle
  mv "$H3_SHARED/oracle.tar.pending" "$H3_SHARED/oracle.tar"
fi
if [ ! -f "$H3_SHARED/baseline-fullgate.tar" ]; then
  tar -cf "$H3_SHARED/baseline-fullgate.tar.pending" -C results/baseline .
  mv "$H3_SHARED/baseline-fullgate.tar.pending" "$H3_SHARED/baseline-fullgate.tar"
fi
python3 "$H3_JUDGE/worker.py" --stage "$1" --shared "$H3_SHARED/results" --oracle "$1/oracle" --oracle-sha256 "$H3_ORACLE_HASH" --limit 100
RUN
