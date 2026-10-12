#!/bin/bash
#SBATCH --job-name=h3-live-benchmark
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cj/h3-live/benchmark-%j.log
# Usage: sbatch benchmark.sbatch BUILD_CACHE_TAR BENCHMARK_OVERLAY_TAR
# The cache was built at this path; retain it for object dependencies/RPATHs.
set -euo pipefail
LIVE_STAGE=/var/tmp/h3-live-15182
mkdir -p "$LIVE_STAGE"
tar -xf "$1" -C "$LIVE_STAGE"
tar -xf "$2" -C "$LIVE_STAGE"
export VSA=1 H3_LIVE_BENCHMARK=1
bash "$LIVE_STAGE/tests/live/resident.sbatch" "$2" 4 "$LIVE_STAGE"
