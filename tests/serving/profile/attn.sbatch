#!/bin/bash
#SBATCH --job-name=h3-attn
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cj/h3-serve/attn-%j.log
# FlashAttention-2 adapter versus FlashAttention-3 on DiT shapes.
# Usage: sbatch attn.sbatch SOURCE_TAR
set -euo pipefail
SOURCE=$1
STAGE=/var/tmp/h3-attn-$SLURM_JOB_ID
mkdir -p "$STAGE/.deps"
tar -xf "$SOURCE" -C "$STAGE"
tar -xzf /home/cj/h3-cuda/flash-deps.tar.gz -C "$STAGE/.deps"
tar -xzf /home/cj/h3-serve/flash3-deps.tar.gz -C "$STAGE/.deps"
apptainer exec --nv --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$STAGE" <<'RUN'
set -euo pipefail
cd "$1"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm,power.limit --format=csv
mkdir -p build-attn
F2="-I.deps/flash-cuda -I.deps/cutlass-flash/include"
F3="-I.deps/flash3/hopper -I.deps/flash3/cutlass/include"
FA3_DISABLE="-DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_SPLIT -DFLASHATTENTION_DISABLE_PAGEDKV \
 -DFLASHATTENTION_DISABLE_APPENDKV -DFLASHATTENTION_DISABLE_LOCAL -DFLASHATTENTION_DISABLE_SOFTCAP \
 -DFLASHATTENTION_DISABLE_PACKGQA -DFLASHATTENTION_DISABLE_FP16 -DFLASHATTENTION_DISABLE_FP8 \
 -DFLASHATTENTION_DISABLE_VARLEN -DFLASHATTENTION_DISABLE_SM8x -DFLASHATTENTION_DISABLE_HDIM64 \
 -DFLASHATTENTION_DISABLE_HDIM96 -DFLASHATTENTION_DISABLE_HDIM192 -DFLASHATTENTION_DISABLE_HDIM256"
time nvcc -O3 -std=c++17 -arch=sm_90 --expt-relaxed-constexpr --expt-extended-lambda -DUNFUSE_FMA \
    -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_SOFTCAP -Icuda $F2 \
    -c cuda/flash_attention.cu -o build-attn/fa2.o
time nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda \
    --use_fast_math -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 \
    -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG $FA3_DISABLE --resource-usage -Icuda $F3 \
    -c cuda/flash3_attention.cu -o build-attn/fa3.o 2>&1 | grep -v "^$" | tail -20
nvcc -O2 -std=c++17 -arch=sm_90 -Icuda -c tests/serving/profile/attn_bench.cu -o build-attn/bench.o
nvcc -o build-attn/attn_bench build-attn/bench.o build-attn/fa2.o build-attn/fa3.o -lcuda
for s in 7200 15400 29800; do build-attn/attn_bench $s 56; done
RUN
