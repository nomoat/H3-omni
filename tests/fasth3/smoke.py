"""Standalone FastH3 CLI, invalid settings, resident rebind and warm latency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'serving'))
from compare import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = Path('build-cuda/h3').resolve()
    ffprobe = os.environ.get('H3_FFPROBE', 'ffprobe')
    prompt = 'A cat walks through a sunny garden.'
    wide_prompt = 'A small cat walks slowly through a sunny garden.'
    base = [str(binary), '-d', str(args.weights), '-p', prompt, '--width', '256',
            '--height', '256', '--frames', '22', '--seed', '0']
    records = []
    for label, extra in [('wrong_steps', ['--steps', '20']), ('reuse', ['--reuse', '2']),
                         ('layers', ['--layers', '45']), ('conditioning', ['--first-frame', 'missing.png'])]:
        result = subprocess.run(base + extra + ['-o', str(output / 'invalid.mp4')], capture_output=True, text=True)
        assert result.returncode != 0 and not (output / 'invalid.mp4').exists(), label
        records.append(dict(case=label, passed=True, error=result.stderr[-600:]))
    dump = output / 'cli'
    dump.mkdir()
    movie = dump / 'output.mp4'
    started = time.monotonic()
    with (output / 'cli.log').open('w') as log:
        # Deliberately omit --steps: the checkpoint must select four evaluations.
        subprocess.run(base + ['--profile', '-o', str(movie)],
                       env={**os.environ, 'H3_DUMP_DIR': str(dump)},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    cli_seconds = time.monotonic() - started
    assert re.search(r'denoise\s+4/4', (output / 'cli.log').read_text())
    metadata = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames',
        '-show_streams', '-of', 'json', str(movie)]))
    video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
    assert (video['width'], video['height'], int(video['nb_read_frames'])) == (256, 256, 22)
    assert int(audio['sample_rate']) == 32000
    records.append(dict(case='CLI automatic four steps and MP4/audio', passed=True, seconds=cli_seconds,
                        video=video, audio=audio))
    requests = output / 'requests.txt'
    requests.write_text(f'short|256|256|22|4|0|{prompt}\n'
                        f'wide|864|480|22|4|1|{wide_prompt}\n'
                        f'again|256|256|22|4|0|{prompt}\n')
    serving = output / 'serving'
    serving.mkdir()
    with (output / 'serving.jsonl').open('w') as log:
        subprocess.run(['build-cuda/h3_serve_check', str(args.weights), str(serving), str(requests)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ('short', 'again'):
        result = compare(serving / name, dump)
        assert result['all_equal'] and all(result[n]['finite'] for n in result if n.endswith('.f32')), result
        records.append(dict(case=f'resident {name} equals fresh CLI', passed=True, comparison=result))
    benchmark = output / 'benchmark'
    benchmark.mkdir()
    requests = output / 'benchmark-requests.txt'
    requests.write_text(f'cold|864|480|124|4|42|{prompt}\n'
                        f'warm|864|480|124|4|43|{prompt}\n')
    with (output / 'benchmark.jsonl').open('w') as log:
        subprocess.run(['build-cuda/h3_serve_check', str(args.weights), str(benchmark), str(requests)],
                       env={**os.environ, 'H3_SERVE_DUMP': '0'},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    timings = [json.loads(line) for line in (output / 'benchmark.jsonl').read_text().splitlines()
               if line.startswith('{')]
    result = dict(passed=True, cases=records, benchmark=timings,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, benchmark=timings)), flush=True)


if __name__ == '__main__':
    main()
