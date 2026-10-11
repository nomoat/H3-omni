#!/bin/bash
#SBATCH --job-name=h3-fasth3-build-modes
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-h200x1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cj/h3-fasth3/build-modes-%j.log
set -euo pipefail
STAGE=/var/tmp/h3-fasth3
RESULT=/home/cj/h3-fasth3/results/$SLURM_JOB_ID
mkdir -p "$RESULT"
tar -xmf /home/cj/h3-fasth3/source.tar -C "$STAGE"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -f cuda/Makefile -j8 VSA=0 ICU_ROOT="$1/icu/usr" all
cp build-cuda/h3 "$2/h3-no-vsa"
python3 - "$2" <<'PY'
import json, pathlib, subprocess, sys
folder = pathlib.Path(sys.argv[1]) / 'model'
transformer = folder / 'FL2VA/transformer'
transformer.mkdir(parents=True)
manifest = dict(format='h3.c-fasth3-v1', task='t2av', attention='vsa', guidance_scale=1.,
    dmd_denoising_steps=[999,749,500,250], video_shift=12., audio_shift=3., tile_size=64, sparsity=.9)
(transformer / 'h3_fasth3.json').write_text(json.dumps(manifest))
result = subprocess.run(['build-cuda/h3', '-d', str(folder), '--info'], capture_output=True, text=True)
assert result.returncode == 1 and 'VSA=1' in result.stderr, result
(folder.parent / 'no-vsa.json').write_text(json.dumps(dict(passed=True, error=result.stderr), indent=2))
PY
H3_FLASH3=0 H3_VAE_3XTF32=0 python3 tests/pytorch/primitives.py --output "$2/default-primitives.json"
make -f cuda/Makefile -j8 VSA=1 ICU_ROOT="$1/icu/usr" all
python3 tests/fasth3/contract.py --output "$2/contract.json"
python3 tests/fasth3/vsa.py --fastvideo reference --output "$2/vsa.json"
RUN
