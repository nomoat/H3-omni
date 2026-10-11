"""Frozen build and external-clock benchmark helpers for the first campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

OBJECTS = 'h3 h3_host h3_safetensors h3_weights h3_text_encoder h3_dit_schedule h3_dit h3_video_vae h3_video_encoder h3_audio_vae h3_ffmpeg h3_terminal h3_vision_encoder h3_multimodal'.split()
MODEL = 'h3_host h3_safetensors h3_weights h3_text_encoder h3_dit_schedule h3_dit h3_video_vae h3_audio_vae'.split()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)


def build(root, accepted, output, *, timeout=600):
    root, accepted, output = map(lambda p: Path(p).resolve(), (root, accepted, output))
    output.mkdir(parents=True, exist_ok=True)
    dest = root / 'build-cuda'
    dest.mkdir(exist_ok=True)
    started = time.monotonic()
    # Immutable production objects are reused; only the declared CUDA surface changes.
    for source in (accepted / 'build-cuda').glob('*.o'):
        shutil.copy2(source, dest / source.name)
    shutil.copy2(accepted / 'build-cuda/libh3_tokenizer.so', dest / 'libh3_tokenizer.so')
    cudnn = Path('/opt/conda/lib/python3.11/site-packages/nvidia/cudnn')
    icu = accepted / 'icu/usr'
    libs = ['-lcublas', '-lcublasLt', '-L' + str(cudnn / 'lib'), '-Xlinker=-rpath=' + str(cudnn / 'lib'), '-l:libcudnn.so.9', '-lpthread', '-lm']
    icu_libs = ['-L' + str(icu / 'lib/x86_64-linux-gnu'), '-Xlinker=--disable-new-dtags', '-Xlinker=-rpath=' + str(icu / 'lib/x86_64-linux-gnu'), '-licuuc', '-licui18n']
    cmds = []
    if digest(root / 'cuda/h3_gpu_cuda.cu') != digest(accepted / 'cuda/h3_gpu_cuda.cu'):
        cmds.append(['nvcc', '-O2', '-lineinfo', '-std=c++17', '-arch=sm_90', '--fmad=false', '-Xcompiler=-fPIC', '-I' + str(root), '-I' + str(root / 'cuda'), '-I' + str(cudnn / 'include'), '-c', str(root / 'cuda/h3_gpu_cuda.cu'), '-o', str(dest / 'h3_gpu_cuda.o')])
    gpu = [str(dest / (s + '.o')) for s in ('h3_gpu_cuda', 'flash_attention')]
    cmds.append(['nvcc', '-shared', '-o', str(dest / 'libh3_cuda.so'), *gpu, *libs])
    cmds.append(['nvcc', '-shared', '-o', str(dest / 'libh3_model.so'), *[str(dest / (s + '.o')) for s in MODEL], *gpu, str(dest / 'model_bridge.o'), *libs])
    full = [str(dest / (s + '.o')) for s in OBJECTS] + gpu + [str(dest / 'h3_tokenizer_portable.o')]
    cmds.append(['nvcc', '-o', str(dest / 'h3'), *full, *[str(dest / (s + '.o')) for s in ('main', 'h3_cli', 'linenoise')], *libs, *icu_libs])
    with (output / 'build.log').open('w') as log:
        for cmd in cmds:
            log.write(json.dumps(cmd) + '\n'); log.flush()
            subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    info = dict(passed=True, seconds=time.monotonic()-started,
                source_sha256=digest(root / 'cuda/h3_gpu_cuda.cu'),
                binaries={p.name:digest(p) for p in (dest/'h3',dest/'libh3_cuda.so',dest/'libh3_model.so',dest/'libh3_tokenizer.so')},
                reused_objects={s:digest(dest/(s+'.o')) for s in OBJECTS + ['flash_attention','h3_tokenizer_portable','model_bridge']})
    write_json(output / 'build.json', info)
    return info


def media_check(path):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    frames = sum(1 for _ in reader)
    audio = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path), '-map', '0:a:0', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], capture_output=True, timeout=60, check=True).stdout
    assert frames == 56 and tuple(meta['size']) == (864, 480) and len(audio) > 0
    return dict(frames=frames, size=meta['size'], audio_bytes=len(audio), sha256=digest(path))


def benchmark(root, weights, prompt, output, samples=3):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_prompt = Path(prompt).read_text().strip()
    assert '\t' not in request_prompt and '\n' not in request_prompt
    records = []
    all_started = time.monotonic()
    for index in range(samples):
        target = output / f'request-{index}.mp4'
        cmd = [str(root/'build-cuda/h3'), '-d', str(Path(weights).resolve().parent),
               '-p', request_prompt, '--width','864','--height','480','--frames','56',
               '--steps','20','--layers','50','--reuse','1','--core-reuse','1',
               '--seed','7','-o',str(target)]
        with (output/f'native-{index}.log').open('w') as log:
            started = time.monotonic()
            subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            elapsed = time.monotonic() - started
        record = dict(index=index,seed=7,prompt_sha256=hashlib.sha256(request_prompt.encode()).hexdigest(),seconds=elapsed,command=cmd,media=media_check(target))
        records.append(record)
        write_json(output/'samples.json',records)
        print(json.dumps(dict(event='benchmark_sample',output=str(output),index=index,seconds=elapsed)),flush=True)
    times = [r['seconds'] for r in records]
    score = statistics.median(times)
    report = dict(passed=True,score_seconds=score,samples=times,mad_seconds=statistics.median(abs(t-score) for t in times),mode='new_process_fixed_prompt_fixed_seed_complete_mp4',warmups=0,total_seconds=time.monotonic()-all_started,records=records,file_cache_policy='No global page-cache eviction on the shared node; filesystem pages may remain cached. No process, model or CUDA state survives between samples.')
    write_json(output/'benchmark.json',report)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('root');b.add_argument('accepted');b.add_argument('output')
    b=sub.add_parser('benchmark');b.add_argument('root');b.add_argument('weights');b.add_argument('prompt');b.add_argument('output');b.add_argument('--samples',type=int,default=3)
    args=vars(p.parse_args());command=args.pop('command')
    result=globals()[command](**args)
    print(json.dumps(result),flush=True)
