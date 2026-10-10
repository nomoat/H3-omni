#!/bin/bash
set -euo pipefail
H3_RELEASE_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
exec apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    /data0/apptainer/pytorch_291_cu128.sif \
    bash -c 'export LD_LIBRARY_PATH="$1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; export H3_FFMPEG="$1/bin/ffmpeg"; exec "$1/h3" "${@:2}"' bash "$H3_RELEASE_ROOT" "$@"
