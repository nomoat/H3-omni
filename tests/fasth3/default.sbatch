#!/bin/bash
#SBATCH --job-name=h3-fasth3-default
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-h200x1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:40:00
#SBATCH --output=/home/cj/h3-fasth3/default-%j.log
set -euo pipefail
STAGE=/var/tmp/h3-fasth3
RESULT=/home/cj/h3-fasth3/results/$SLURM_JOB_ID
mkdir -p "$RESULT"
test -f /data0/models/h3-fasth3/vsa-native/FL2VA/transformer/h3_fasth3.json
tar -xmf /home/cj/h3-fasth3/source.tar -C "$STAGE"
cp /home/cj/h3-fasth3/source.tar "$RESULT/source.tar"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" "$RESULT" <<'RUN'
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONPATH="$1/python"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$1/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FLASH3=1 H3_VAE_3XTF32=1
make -f cuda/Makefile -j8 VSA=1 ICU_ROOT="$1/icu/usr" all
python3 tests/fasth3/contract.py --output "$2/contract.json"
python3 tests/fasth3/parity.py --source /data0/models/h3-fasth3/vsa-source \
    --fastvideo reference --oracle hopper --weights /data0/models/h3-fasth3/vsa-native/FL2VA \
    --diffusers "$1/diffusers" --text /home/cj/h3-cuda/results/14274/text/text-1.safetensors \
    --width 512 --height 512 --frames 22 --seed 1 --output "$2/parity"
python3 tests/pytorch/decoders.py --weights /data0/models/h3-fasth3/vsa-native/FL2VA \
    --diffusers "$1/diffusers" --latents "$2/parity/latents.safetensors" --output "$2/decoded"
RUN
