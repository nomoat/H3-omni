#!/bin/bash
#SBATCH --job-name=h3-research-baseline
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-study/baseline-%j.log
set -euo pipefail
H3_STUDY=/home/cj/h3-cuda/autoresearch-study
H3_STAGE=/var/tmp/h3-research-$SLURM_JOB_ID
mkdir -p "$H3_STAGE" "$H3_STUDY/results/$SLURM_JOB_ID"
cp "$H3_STUDY/baseline.c" "$H3_STUDY/prompt-a.txt" "$H3_STUDY/prompt-b.txt" "$H3_STAGE/"
trap 'cp -a "$H3_STAGE/." "$H3_STUDY/results/$SLURM_JOB_ID/"' EXIT
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp /data0/apptainer/pytorch_291_cu128.sif bash -s -- "$H3_STAGE" <<'RUN'
set -euo pipefail
cd "$1"
H3_ACCEPTED=/var/tmp/h3-cuda-14274
H3_CUDNN=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn
export OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=1
export PYTHONPATH="$H3_ACCEPTED/python"
export LD_LIBRARY_PATH="$H3_ACCEPTED/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
cc -O2 -std=c11 -I"$H3_ACCEPTED" -c baseline.c -o baseline.o
H3_OBJECTS=()
for name in h3 h3_host h3_safetensors h3_weights h3_text_encoder h3_dit_schedule h3_dit h3_video_vae h3_video_encoder h3_audio_vae h3_ffmpeg h3_terminal h3_vision_encoder h3_multimodal h3_gpu_cuda flash_attention h3_tokenizer_portable; do
  H3_OBJECTS+=("$H3_ACCEPTED/build-cuda/$name.o")
done
nvcc -o baseline baseline.o "${H3_OBJECTS[@]}" -lcublas -lcublasLt -L"$H3_CUDNN/lib" -Xlinker=-rpath="$H3_CUDNN/lib" -l:libcudnn.so.9 -lpthread -lm -L"$H3_ACCEPTED/icu/usr/lib/x86_64-linux-gnu" -Xlinker=-rpath="$H3_ACCEPTED/icu/usr/lib/x86_64-linux-gnu" -licuuc -licui18n
sha256sum baseline baseline.c "${H3_OBJECTS[@]}" > source-and-object-sha256.txt
nvidia-smi > device.txt
./baseline /var/tmp/h3-cuda-14228/local-model prompt-a.txt prompt-b.txt . > events.jsonl 2> runtime.log
python3 - <<'PY'
import json,subprocess
from pathlib import Path
import imageio_ffmpeg
rows=[]
for path in sorted(Path('.').glob('request-*.mp4')):
    reader=imageio_ffmpeg.read_frames(str(path));meta=next(reader);frames=sum(1 for _ in reader)
    audio=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-'],check=True,capture_output=True).stdout
    assert frames==56 and tuple(meta['size'])==(864,480) and len(audio)>0
    rows.append(dict(file=str(path),frames=frames,size=meta['size'],decoded_audio_bytes=len(audio)))
assert len(rows)==3
Path('media-check.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2)+'\n')
for line in Path('events.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['event']=='result':print(json.dumps(r),flush=True)
PY
RUN
