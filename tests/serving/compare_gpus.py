"""Compare multi-GPU serving outputs (H3_GPUS=N) with one-GPU outputs.

Both directories hold one subdirectory per request with the dumps written by
h3_serve_check (video_latent.f32, audio_latent.f32, audio_pcm.f32,
video_rgb.f32). Reports, per file: bit equality, relative L2, maximum
absolute difference and the fraction of differing values, plus for the RGB
dump the fraction of 8-bit pixels that are equal.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FILES = ['video_latent.f32', 'audio_latent.f32', 'audio_pcm.f32', 'video_rgb.f32']


def compare(a_path, b_path):
    a = np.fromfile(a_path, dtype=np.float32).astype(np.float64)
    b = np.fromfile(b_path, dtype=np.float32).astype(np.float64)
    if a.shape != b.shape:
        return dict(shape_mismatch=[a.size, b.size])
    d = a - b
    entry = dict(equal=bool((d == 0).all()), finite=bool(np.isfinite(a).all()),
                 relative_l2=float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-30)),
                 max_abs=float(np.abs(d).max()), differing=float((d != 0).mean()))
    if a_path.name == 'video_rgb.f32':
        ea = np.round(np.clip(a, 0, 1) * 255)
        eb = np.round(np.clip(b, 0, 1) * 255)
        entry['pixels_equal'] = float((ea == eb).mean())
        entry['max_8bit'] = int(np.abs(ea - eb).max())
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--one', required=True, help='H3_GPUS=1 serve directory')
    parser.add_argument('--many', required=True, help='H3_GPUS=N serve directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    one, many = Path(args.one), Path(args.many)
    report = {}
    for request in sorted(p.name for p in many.iterdir() if p.is_dir()):
        if not (one / request).is_dir():
            continue
        report[request] = {name: compare(many / request / name, one / request / name)
                           for name in FILES
                           if (many / request / name).exists() and (one / request / name).exists()}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    for request, files in report.items():
        for name, entry in files.items():
            print(request, name, json.dumps(entry))


if __name__ == '__main__':
    main()
