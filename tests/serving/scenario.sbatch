#!/bin/bash
#SBATCH --job-name=h3-scenario
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cj/h3-serve/scenario-%j.log
# Differential check of a new build against an accepted build on one request
# outside the serving test set (canvas, duration, conditioning). Both builds
# run the same one-shot request in the reference mode and the default mode.
#   reference mode: every dump and the MP4 must be bit-identical;
#   default mode: latents and audio bit-identical, decoded RGB reported.
# Usage: sbatch scenario.sbatch NEW_TAR OLD_TAR SCENARIO
#   SCENARIO: landscape768 | portrait768 | long15s | firstlast | refimage | refvideo
set -euo pipefail
NEW=$1
OLD=$2
SCENARIO=$3
STAGE=/var/tmp/h3-scenario-$SLURM_JOB_ID
RESULT=/home/cj/h3-serve/results/$SLURM_JOB_ID
mkdir -p "$STAGE/new/.deps" "$STAGE/old/.deps" "$STAGE/icu" "$RESULT"
tar -xf "$NEW" -C "$STAGE/new"
tar -xf "$OLD" -C "$STAGE/old"
for tree in new old; do
    tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/$tree/.deps"
    tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/$tree/.deps"
done
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
if [ -f /var/tmp/h3-cuda-14228/local-model/FL2VA/transformer/config.json ]; then MODEL=/var/tmp/h3-cuda-14228/local-model; fi
echo "node $(hostname) scenario $SCENARIO"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true; find "$RESULT" -name "*.f32" -delete' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp --bind /home/cj:/home/cj \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" "$SCENARIO" <<'RUN'
set -euo pipefail
cd "$1"
MODEL="$2"
SCENARIO="$3"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
for tree in new old; do
    make -C "$1/$tree" -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/$tree/build-cuda/h3" \
        > "$1/build-$tree.log" 2>&1 || { tail -30 "$1/build-$tree.log"; exit 1; }
done
IN=/home/cj/h3-serve/inputs
FOX="A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind."
case "$SCENARIO" in
  landscape768) ARGS=(--width 1344 --height 768 --frames 56); PROMPT="$FOX" ;;
  portrait768)  ARGS=(--width 768 --height 1344 --frames 56); PROMPT="A waterfall falls into a green pool in a tropical forest. Vertical shot, mist, birds and the roar of water." ;;
  long15s)      ARGS=(--width 864 --height 480 --frames 362); PROMPT="$FOX" ;;
  firstlast)    ARGS=(--width 864 --height 480 --frames 56 --first-frame "$IN/fox-first.png" --last-frame "$IN/fox-last.png"); PROMPT="The red fox keeps walking through the snow from left to right." ;;
  refimage)     ARGS=(--width 864 --height 480 --frames 56 --ref-image "$IN/guitar-ref.png"); PROMPT="The street musician shown in Picture 1 plays a slow song while people walk past at dusk." ;;
  refvideo)     ARGS=(--width 864 --height 480 --frames 56 --ref-video "$IN/boat-ref.mp4"); PROMPT="The wooden boat from the reference video drifts closer to the shore as the sun rises higher." ;;
  *) echo "unknown scenario"; exit 2 ;;
esac
mkdir -p results
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > results/gpu-memory.txt &
SAMPLER=$!
for mode in reference default; do
    for tree in old new; do
        out="results/$mode/$tree"
        mkdir -p "$out"
        if [ $mode = reference ]; then export H3_FLASH3=0 H3_VAE_3XTF32=0; else unset H3_FLASH3 H3_VAE_3XTF32; fi
        start=$(date +%s.%N)
        if H3_DUMP_DIR="$out" "$1/$tree/build-cuda/h3" -d "$MODEL" -p "$PROMPT" "${ARGS[@]}" --steps 20 \
            --layers 50 --reuse 1 --core-reuse 1 --seed 7 -o "$out/output.mp4" > "$out/stdout.log" 2> "$out/stderr.log"; then
            status=ok
        else
            status="FAILED($?)"; tail -5 "$out/stderr.log"
        fi
        end=$(date +%s.%N)
        echo "{\"mode\":\"$mode\",\"build\":\"$tree\",\"status\":\"$status\",\"seconds\":$(python3 -c "print(round($end-$start,1))")}"
    done
done
kill $SAMPLER || true
echo "peak GPU memory MiB: $(sort -n results/gpu-memory.txt | tail -1)"
ls results/*/*/
python3 "$1/new/tests/serving/compare.py" --serve results/reference --fresh results/reference --pairs new=old \
    --output results/reference-compare.json && echo "REFERENCE MODE BIT-IDENTICAL: yes" || echo "REFERENCE MODE BIT-IDENTICAL: NO"
python3 "$1/new/tests/serving/compare.py" --serve results/default --fresh results/default --pairs new=old \
    --output results/default-compare.json > /dev/null || true
python3 - <<'EOF'
import json
d = json.load(open("results/default-compare.json"))["new"]
names = ["video_latent.f32", "audio_latent.f32", "audio_pcm.f32", "video_rgb.f32", "output.mp4"]
print("default mode equal:", {n: d[n].get("equal") for n in names})
print("DEFAULT LATENTS+AUDIO BIT-IDENTICAL:", "yes" if all(d[n].get("equal") for n in names[:3]) else "NO")
EOF
python3 "$1/new/tests/serving/rgb_diff.py" --a results/default --b results/default --pairs new=old
RUN
