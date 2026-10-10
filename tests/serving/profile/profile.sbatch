#!/bin/bash
#SBATCH --job-name=h3-profile
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cj/h3-serve/profile-%j.log
# CUPTI kernel trace of warm resident requests at 864x480, 20 steps.
# Usage: sbatch profile.sbatch SOURCE_TAR SET   (SET: s10 | short | api)
# "api" also records host CUDA API calls and attributes GPU idle time.
set -euo pipefail
ROOT=/home/cj/h3-serve
SOURCE=$1
SET=$2
STAGE=/var/tmp/h3-profile-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$RESULT"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
LOCAL_MODEL=/var/tmp/h3-cuda-14228/local-model
if [ -f "$LOCAL_MODEL/FL2VA/transformer/config.json" ]; then MODEL=$LOCAL_MODEL; fi
echo "node $(hostname) model $MODEL set $SET"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$SET" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"
SET="$3"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export H3_SERVE_DUMP=0
mkdir -p results/serve
nvidia-smi > results/device.txt
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check"
CUPTI=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_cupti
gcc -O2 -shared -fPIC -I"$CUPTI/include" -I/usr/local/cuda/include \
    tests/serving/profile/cupti_trace.c -o build-cuda/libcupti_trace.so \
    -L"$CUPTI/lib" -l:libcupti.so.12 -Wl,-rpath,"$CUPTI/lib" -lpthread
FOX="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
BOAT="A small wooden fishing boat moves slowly across a calm mountain lake at sunrise. Mist drifts between pine trees, gentle waves touch the rocks, and birds call in the distance."
case "$SET" in
  s10) REQ="warmup|864|480|56|20|1|$FOX
p10|864|480|240|20|7|$BOAT" ; NAMES=p10 ;;
  short) REQ="warmup|864|480|56|20|1|$FOX
p2|864|480|56|20|7|$BOAT
p5|864|480|120|20|7|$BOAT" ; NAMES=p2,p5 ;;
  api) REQ="warmup|864|480|56|20|1|$FOX
p2|864|480|56|20|7|$BOAT" ; NAMES=p2 ; export H3_CUPTI_API=1 ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
echo "$REQ" > results/requests.txt
CUDA_INJECTION64_PATH="$1/build-cuda/libcupti_trace.so" H3_CUPTI_OUT=results/trace.txt \
    build-cuda/h3_serve_check "$MODEL" results/serve results/requests.txt \
    > results/serve/events.jsonl 2> results/serve/stderr.log
cat results/serve/events.jsonl
ls -la results/trace.txt
python3 tests/serving/profile/report.py --trace results/trace.txt \
    --events results/serve/events.jsonl --requests "$NAMES" \
    --output results/profile.json | tee results/profile.txt
if [ "$SET" = api ]; then
    for phase in "denoise|audio VAE" "video VAE load|FFmpeg"; do
        python3 tests/serving/profile/host_gaps.py --trace results/trace.txt \
            --events results/serve/events.jsonl --request p2 \
            --phase "${phase%%|*}" --next-phase "${phase##*|}" | tee -a results/host-gaps.txt
    done
fi
gzip results/trace.txt
RUN
