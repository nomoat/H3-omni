"""Compare resident-serving outputs with fresh one-shot outputs bit for bit.

Each serving request directory holds video_latent.f32, audio_latent.f32,
audio_pcm.f32, video_rgb.f32 and output.mp4. A request repeated later in the
serving sequence is compared with the same fresh reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FILES = ['video_latent.f32', 'audio_latent.f32', 'audio_pcm.f32',
         'video_rgb.f32', 'output.mp4']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(serve, fresh):
    result = {}
    for name in FILES:
        a, b = serve / name, fresh / name
        if not a.exists() or not b.exists():
            result[name] = dict(equal=False, missing=[str(p) for p in (a, b) if not p.exists()])
            continue
        entry = dict(equal=digest(a) == digest(b), bytes=a.stat().st_size)
        if name.endswith('.f32'):
            x = np.fromfile(a, dtype=np.float32)
            y = np.fromfile(b, dtype=np.float32)
            entry['finite'] = bool(np.isfinite(x).all())
            if x.shape == y.shape and not entry['equal']:
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                entry['max_abs'] = float(d.max())
                entry['differing'] = int((x != y).sum())
        result[name] = entry
    result['all_equal'] = all(result[n].get('equal') for n in FILES)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', required=True)
    parser.add_argument('--fresh', required=True)
    parser.add_argument('--pairs', required=True,
                        help='comma list of serve=fresh request names')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {}
    for pair in args.pairs.split(','):
        serve, fresh = pair.split('=')
        report[serve] = dict(reference=fresh, **compare(
            Path(args.serve) / serve, Path(args.fresh) / fresh))
    report['passed'] = all(v['all_equal'] for v in report.values())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'],
                          requests={k: v['all_equal'] for k, v in report.items()
                                    if k != 'passed'})))


if __name__ == '__main__':
    main()
