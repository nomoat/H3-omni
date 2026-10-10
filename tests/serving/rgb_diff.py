"""Decoded-video difference between two runs with identical latents.

Reports float relative L2 and maximum difference of video_rgb.f32, and the
8-bit pixel difference after the same clamp and rounding used for encoding
([-1, 1] or [0, 1] range is detected from the data)."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--pairs', required=True, help='comma list of a_name=b_name')
    args = parser.parse_args()
    report = {}
    for pair in args.pairs.split(','):
        left, right = pair.split('=')
        x = np.fromfile(Path(args.a) / left / 'video_rgb.f32', dtype=np.float32).astype(np.float64)
        y = np.fromfile(Path(args.b) / right / 'video_rgb.f32', dtype=np.float32).astype(np.float64)
        lo = -1.0 if min(x.min(), y.min()) < -0.05 else 0.0
        to8 = lambda v: np.rint((np.clip(v, lo, 1.0) - lo) / (1.0 - lo) * 255.0)
        d8 = np.abs(to8(x) - to8(y))
        report[left] = dict(
            relative_l2=float(np.linalg.norm(x - y) / np.linalg.norm(y)),
            max_abs=float(np.abs(x - y).max()), range=[float(y.min()), float(y.max())],
            pixels_equal=float((d8 == 0).mean()), pixels_off_by_1=float((d8 == 1).mean()),
            pixels_off_by_more=float((d8 > 1).mean()), max_8bit=int(d8.max()))
        print(json.dumps({left: report[left]}))


if __name__ == '__main__':
    main()
