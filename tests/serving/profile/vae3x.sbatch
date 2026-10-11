#!/bin/bash
#SBATCH --job-name=h3-vae3x
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cj/h3-serve/vae3x-%j.log
# Video-VAE 3xTF32 accuracy: (1) PyTorch decoder acceptance on the job-14416
# trajectories, (2) serving with FlashAttention-2 so latents equal job 14416,
# then decoded-RGB and 8-bit differences against the FP32 decoder.
# Usage: sbatch vae3x.sbatch SOURCE_TAR
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/cj/h3-serve
STAGE=/var/tmp/h3-vae3x-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/python" "$STAGE/diffusers" "$STAGE/.deps" "$RESULT"
tar -xf "$1" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
tar -xf /home/cj/h3-cuda/python-deps.tar -C "$STAGE/python"
tar -xf /home/cj/h3-cuda/diffusers.tar -C "$STAGE/diffusers"
SHARED=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
MODEL=$SHARED
if [ -f /var/tmp/h3-cuda-14228/local-model/FL2VA/transformer/config.json ]; then MODEL=/var/tmp/h3-cuda-14228/local-model; fi
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true; find "$RESULT" -name "*.f32" -delete; find "$RESULT" -name "*.safetensors" -size +50M -delete' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$SHARED" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONPATH="$1/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
export H3_VAE_3XTF32=1 H3_FLASH3=0
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" all "$1/build-cuda/h3_serve_check" > results-build.log 2>&1 || { tail -30 results-build.log; exit 1; }
mkdir -p results/serve results/decoded
python3 tests/pytorch/decoder_suite.py --weights "$3/FL2VA" --diffusers "$1/diffusers" \
    --trajectories /home/cj/h3-serve/results/14416/accept/trajectories --output results/decoded \
    2>&1 | grep -v Warning | grep -v WeightNorm | tail -20
BOAT=$(python3 -c 'import json;print(json.load(open("tests/pytorch/extended.json"))["prompt"])')
FOX="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
CAT="A cat walks through a sunny garden."
CITY=$'\xe4\xb8\x80\xe5\x8f\xaa\xe6\xa9\x98\xe8\x89\xb2\xe7\x9a\x84\xe5\xb0\x8f\xe7\x8c\xab\xe5\x9c\xa8\xe4\xb8\x8b\xe9\x9b\xa8\xe7\x9a\x84\xe5\x9f\x8e\xe5\xb8\x82\xe8\xa1\x97\xe9\x81\x93\xe4\xb8\x8a\xe5\xa5\x94\xe8\xb7\x91\xef\xbc\x8c\xe9\x9c\x93\xe8\x99\xb9\xe7\x81\xaf\xe5\x80\x92\xe6\x98\xa0\xe5\x9c\xa8\xe6\xb0\xb4\xe5\x9d\x91\xe9\x87\x8c\xef\xbc\x8c\xe8\xbf\x9c\xe5\xa4\x84\xe4\xbc\xa0\xe6\x9d\xa5\xe6\xb1\xbd\xe8\xbd\xa6\xe7\x9a\x84\xe5\xa3\xb0\xe9\x9f\xb3\xe3\x80\x82'
cat > results/requests.txt <<EOF
r1-boat|864|480|56|20|7|$BOAT
r2-fox|864|480|56|20|11|$FOX
r4-cat|512|512|22|20|0|$CAT
r6-city|256|256|22|12|42|$CITY
EOF
build-cuda/h3_serve_check "$2" results/serve results/requests.txt > results/serve/events.jsonl 2> results/serve/stderr.log
python3 tests/serving/compare.py --serve results/serve --fresh /home/cj/h3-serve/results/14416/fresh \
    --pairs r1-boat=r1-boat,r2-fox=r2-fox,r4-cat=r4-cat,r6-city=r6-city --output results/serve/compare.json || true
python3 -c 'import json;d=json.load(open("results/serve/compare.json"));print({k:{f:v[f]["equal"] for f in ["video_latent.f32","audio_pcm.f32","video_rgb.f32"]} for k,v in d.items() if k!="passed"})'
python3 tests/serving/rgb_diff.py --a results/serve --b /home/cj/h3-serve/results/14416/fresh \
    --pairs r1-boat=r1-boat,r2-fox=r2-fox,r4-cat=r4-cat,r6-city=r6-city | tee results/rgb-diff.jsonl
RUN
