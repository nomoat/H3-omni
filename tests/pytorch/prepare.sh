#!/bin/bash
# Run on the login node, without a GPU allocation.
set -euo pipefail
export PYTHONNOUSERSITE=1
H3_ROOT=/home/cj/h3-cuda
H3_PREP=/var/tmp/h3-reference-prep
mkdir -p "$H3_PREP/diffusers"
tar -xf "$H3_ROOT/diffusers.tar" -C "$H3_PREP/diffusers"
python3 -m venv --system-site-packages "$H3_PREP/venv"
"$H3_PREP/venv/bin/python3" -m pip install \
  "$H3_PREP/diffusers" transformers==5.5.3 accelerate==1.14.0 \
  imageio-ffmpeg
"$H3_PREP/venv/bin/python3" -m pip freeze > "$H3_ROOT/reference-requirements.lock"
"$H3_PREP/venv/bin/python3" -c 'from diffusers import MiniMaxH3Transformer3DModel; import transformers; print("reference imports OK", transformers.__version__)'
tar -cf "$H3_ROOT/python-deps.tar" -C "$H3_PREP/venv/lib/python3.11/site-packages" .
