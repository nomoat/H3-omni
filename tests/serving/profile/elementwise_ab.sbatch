#!/bin/bash
#SBATCH --job-name=h3-elem-ab
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cj/h3-serve/elementwise-ab-%j.log
# Same-GPU A/B of the DiT elementwise kernels in the default mode: generated
# kernels (H3_GENERATED_ELEMENTWISE=1) against the hand-written fast kernels,
# two warm 10 s requests each, with SM clock and power every 100 ms. Reports
# request and denoise seconds and the clock/power medians inside denoise.
# Usage: sbatch elementwise_ab.sbatch SOURCE_TAR
set -euo pipefail
STAGE=/var/tmp/h3-elem-ab-$SLURM_JOB_ID
RESULT=/home/cj/h3-serve/results/$SLURM_JOB_ID
mkdir -p "$STAGE/icu" "$STAGE/.deps" "$RESULT"
tar -xf "$1" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps-v2.tar.gz -C "$STAGE/.deps"
tar -xf /home/cj/h3-cuda/icu.tar -C "$STAGE/icu"
MODEL=/data0/models/h3-cuda/42ed227ee7df40d41602854ae760620d6eb651fe
if [ -f /var/tmp/h3-cuda-14228/local-model/FL2VA/transformer/config.json ]; then MODEL=/var/tmp/h3-cuda-14228/local-model; fi
echo "node $(hostname) model $MODEL"
trap 'cp -a "$STAGE/results/." "$RESULT/" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$MODEL" <<'RUN'
set -euo pipefail
cd "$1"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg H3_SERVE_DUMP=0
make -f cuda/Makefile -j"$OMP_NUM_THREADS" ICU_ROOT="$1/icu/usr" "$1/build-cuda/h3_serve_check" > build.log 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit,enforced.power.limit --format=csv
BOAT="A small wooden fishing boat moves slowly across a calm mountain lake at sunrise. Mist drifts between pine trees, gentle waves touch the rocks, and birds call in the distance."
printf 'warmup|864|480|56|20|1|%s\np10a|864|480|240|20|7|%s\np10b|864|480|240|20|7|%s\n' "$BOAT" "$BOAT" "$BOAT" > requests.txt
mkdir -p results
python3 -c 'import time; print(time.time() - time.monotonic())' > results/monotonic_offset.txt
for mode in generated fast; do
    mkdir -p results/$mode
    if [ $mode = generated ]; then export H3_GENERATED_ELEMENTWISE=1; else unset H3_GENERATED_ELEMENTWISE; fi
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
        --format=csv,noheader,nounits -lms 100 > results/$mode/clocks.csv &
    S=$!
    build-cuda/h3_serve_check "$2" results/$mode requests.txt > results/$mode/events.jsonl 2> results/$mode/stderr.log
    kill $S
done
python3 - <<'EOF' | tee results/summary.txt
import json, datetime
offset = float(open("results/monotonic_offset.txt").read())
for mode in ("generated", "fast"):
    rows = []
    for line in open(f"results/{mode}/clocks.csv"):
        p = [x.strip() for x in line.split(",")]
        try:
            t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((t, float(p[1]), float(p[3]), float(p[4])))
        except Exception:
            pass
    for line in open(f"results/{mode}/events.jsonl"):
        e = json.loads(line)
        if e.get("event") != "request" or not e["name"].startswith("p10"):
            continue
        ph = {x["phase"]: x for x in e["phases"]}
        d0 = e["start"] + ph["denoise"]["first"] + offset
        d1 = e["start"] + ph["denoise"]["last"] + offset
        inside = [r for r in rows if d0 <= r[0] <= d1]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
        print(f"{mode:9s} {e['name']} request {e['seconds']:.1f} s denoise {d1 - d0:.2f} s "
              f"denoise samples {len(inside)} SM MHz median {med([r[1] for r in inside])} "
              f"power median {med([r[2] for r in inside])} W")
EOF
RUN
