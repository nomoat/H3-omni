#!/bin/bash
#SBATCH --job-name=h3-search-compile
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-100/compile-%j.log
set -euo pipefail
H3_STAGE=/var/tmp/h3-search-100
H3_SHARED=/home/cj/h3-cuda/autoresearch-100
mkdir -p "$H3_STAGE/compile-smoke" "$H3_SHARED/compile-smoke"
tar -xf "$H3_SHARED/inputs.tar" -C "$H3_STAGE"
trap 'cp -a "$H3_STAGE/compile-smoke/." "$H3_SHARED/compile-smoke/"' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/var/tmp/h3-cuda-14274/diffusers/src:/var/tmp/h3-cuda-14274/python
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/var/tmp/h3-cuda-14274/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
python3 - <<'PY'
from pathlib import Path
import shutil,subprocess,sys
sys.path.insert(0,'harness')
from runtime import build
for n in (1,12):
    target=Path(f'candidates/{n:04d}')
    if not target.exists():shutil.copytree('seed',target,ignore=shutil.ignore_patterns('build-cuda','research','__pycache__'))
    shutil.copy2(f'sources/{n:04d}.cu',target/'cuda/h3_gpu_cuda.cu')
    build(target,'/var/tmp/h3-cuda-14274',f'compile-smoke/{n:04d}')
    subprocess.run([sys.executable,str(Path('seed/tests/pytorch/primitives.py').resolve()),'--library',str((target/'build-cuda/libh3_cuda.so').resolve()),'--output',str(Path(f'compile-smoke/{n:04d}/primitives.json').resolve())],check=True,cwd=target)
PY
RUN
