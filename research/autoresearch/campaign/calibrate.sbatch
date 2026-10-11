#!/bin/bash
#SBATCH --job-name=h3-search-judge
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-100/judge-%j.log
set -euo pipefail
H3_STAGE=/var/tmp/h3-search-100
H3_SHARED=/home/cj/h3-cuda/autoresearch-100
H3_CALIBRATION="$H3_STAGE/calibration-$SLURM_JOB_ID"
mkdir -p "$H3_STAGE/seed/research/autoresearch/campaign" "$H3_CALIBRATION" "$H3_SHARED/calibration/$SLURM_JOB_ID"
tar -xf "$H3_SHARED/harness.tar" -C "$H3_STAGE/seed/research/autoresearch/campaign"
trap 'cp -a "$H3_CALIBRATION/." "$H3_SHARED/calibration/$SLURM_JOB_ID/"; cp "$H3_STAGE/oracle/verification.json" "$H3_SHARED/calibration/$SLURM_JOB_ID/oracle-prepare.json" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" "$H3_CALIBRATION" <<'RUN'
set -euo pipefail
cd "$1"
H3_CALIBRATION="$2"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/var/tmp/h3-cuda-14274/diffusers/src:/var/tmp/h3-cuda-14274/python
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/var/tmp/h3-cuda-14274/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
H3_JUDGE="$1/seed/research/autoresearch/campaign"
if [ -d oracle ]; then mv oracle "oracle-before-$SLURM_JOB_ID"; fi
if [ ! -d unit-oracle ]; then
  python3 "$H3_JUDGE/frozen_primitives.py" capture --tests-dir "$1/seed/tests/pytorch" --output-dir "$1/unit-oracle" > "$H3_CALIBRATION/unit-capture.json"
fi
python3 "$H3_JUDGE/verify.py" prepare --reference-root /home/cj/h3-cuda/results/14274 --diffusers /var/tmp/h3-cuda-14274/diffusers --unit-oracle "$1/unit-oracle" --output "$1/oracle" > "$H3_CALIBRATION/prepare.log"
H3_ORACLE_HASH=$(python3 -c 'import json;print(json.load(open("oracle/verification.json"))["manifest_sha256"])')
cp oracle/verification.json "$H3_CALIBRATION/oracle-prepare.json"
cp oracle/manifest.json "$H3_CALIBRATION/oracle-manifest.json"
python3 "$H3_JUDGE/verify.py" promote --oracle "$1/oracle" --oracle-sha256 "$H3_ORACLE_HASH" --candidate-root "$1/seed" --weights /var/tmp/h3-cuda-14228/local-model/FL2VA --diffusers /var/tmp/h3-cuda-14274/diffusers --output "$1/results/baseline/fullgate" > "$H3_CALIBRATION/seed-gate.log"
cp results/baseline/fullgate/verification.json "$H3_CALIBRATION/seed-verification.json"
RUN
