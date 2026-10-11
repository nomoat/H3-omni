#!/bin/bash
#SBATCH --job-name=h3-mergecheck
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cj/h3-serve/mergecheck-%j.log
# Default-mode check for a change whose only value change is in the video VAE
# decode: latents and audio must equal an earlier default-mode fresh run bit
# for bit; decoded RGB is compared with that run and with the FP32 reference.
# Usage: sbatch mergecheck.sbatch SOURCE_TAR [DEFAULT_FRESH [FP32_FRESH]]
set -euo pipefail
ROOT=/home/cj/h3-serve
DEFAULT_REF=${2:-$ROOT/results/14438/fresh}
FP32_REF=${3:-$ROOT/results/14416/fresh}
STAGE=/var/tmp/h3-mergecheck-$SLURM_JOB_ID
RESULT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$RESULT"
tar -xf "$1" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
if [ -f /var/tmp/h3-cuda-14228/local-model/FL2VA/transformer/config.json ]; then MODEL=/var/tmp/h3-cuda-14228/local-model; fi
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true; find "$RESULT" -name "*.f32" -delete' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$DEFAULT_REF" "$FP32_REF" <<'RUN'
set -euo pipefail
cd "$1"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check" > build.log 2>&1 || { tail -30 build.log; exit 1; }
BOAT=$(python3 -c 'import json;print(json.load(open("tests/pytorch/extended.json"))["prompt"])')
FOX="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
CAT="A cat walks through a sunny garden."
CITY="一只橘色的小猫在下雨的城市街道上奔跑，霓虹灯倒映在水坑里，远处传来汽车的声音。"
mkdir -p results/serve
cat > results/requests.txt <<EOF
r1-boat|864|480|56|20|7|$BOAT
r2-fox|864|480|56|20|11|$FOX
r4-cat|512|512|22|20|0|$CAT
r6-city|256|256|22|12|42|$CITY
EOF
build-cuda/h3_serve_check "$2" results/serve results/requests.txt > results/serve/events.jsonl 2> results/serve/stderr.log
python3 tests/serving/compare.py --serve results/serve --fresh "$3" \
    --pairs r1-boat=r1-boat,r2-fox=r2-fox,r4-cat=r4-cat,r6-city=r6-city --output results/serve/compare.json || true
python3 - <<'EOF'
import json
d = json.load(open("results/serve/compare.json"))
names = ["video_latent.f32", "audio_latent.f32", "audio_pcm.f32", "video_rgb.f32", "output.mp4"]
ok = True
for k, v in d.items():
    if k == "passed": continue
    eq = {n: v[n]["equal"] for n in names}
    print(k, eq)
    ok &= eq["video_latent.f32"] and eq["audio_latent.f32"] and eq["audio_pcm.f32"]
print(json.dumps({"latents_and_audio_bit_identical": ok}))
EOF
echo "== RGB vs previous default mode"
python3 tests/serving/rgb_diff.py --a results/serve --b "$3" --pairs r1-boat=r1-boat,r2-fox=r2-fox,r4-cat=r4-cat,r6-city=r6-city
echo "== RGB vs FP32 reference"
python3 tests/serving/rgb_diff.py --a results/serve --b "$4" --pairs r1-boat=r1-boat,r2-fox=r2-fox,r4-cat=r4-cat,r6-city=r6-city
RUN
