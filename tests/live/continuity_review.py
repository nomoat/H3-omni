"""Review raw emitted frames/PCM at request and publication boundaries."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

scratch, output = map(Path, sys.argv[1:])
width, height, count, rate, fps = 864, 480, 1195, 32000, 24
samples = round(count * rate / fps)
rgb = np.memmap(scratch / 'published.rgb24', np.uint8, mode='r', shape=(count, height, width, 3))
pcm = np.memmap(scratch / 'published-pcm.f32', np.float32, mode='r', shape=(2, samples))
diff = np.array([np.abs(rgb[i].astype(np.float32) - rgb[i-1]).mean() for i in range(1, count)])
rows = [json.loads(line) for line in (output / 'continuity.jsonl').read_text().splitlines()]
report = {'frames': count, 'fps': fps, 'adjacent_rgb_mae_median': float(np.median(diff)),
          'adjacent_rgb_mae_p95': float(np.percentile(diff, 95)), 'seams': []}
for kind, boundaries in [('request', [5+119*i for i in range(1, 10)]),
                         ('publication', [r['published_end'] for r in rows if 'request' in r][:-1])]:
    for page in range(3):
        canvas = Image.new('RGB', (4*324, 3*204), '#202020')
        draw = ImageDraw.Draw(canvas)
        for row, boundary in enumerate(boundaries[page*3:page*3+3]):
            for col, offset in enumerate([-6, -1, 0, 6]):
                index = boundary + offset
                frame = Image.fromarray(rgb[index]).resize((324, 180))
                canvas.paste(frame, (col*324, row*204+24))
                draw.text((col*324+5, row*204+5), f'{kind} {boundary/fps:.3f}s / frame {index}', fill='white')
        canvas.save(output / f'{kind}-seams-{page+1}.jpg', quality=95)
    for boundary in boundaries:
        sample = round(boundary*rate/fps)
        local = diff[max(0,boundary-25):boundary+24]
        window = pcm[:, sample-1600:sample+1600]
        jumps = np.abs(np.diff(window, axis=1))
        report['seams'].append({'kind': kind, 'frame': boundary, 'time': boundary/fps,
            'rgb_mae': float(diff[boundary-1]), 'local_rgb_mae_median': float(np.median(local)),
            'pcm_jump': float(np.max(np.abs(pcm[:,sample]-pcm[:,sample-1]))),
            'local_pcm_jump_p95': float(np.percentile(jumps,95)),
            'local_pcm_rms': float(np.sqrt(np.mean(window**2)))})
report['pcm_finite'] = bool(np.isfinite(pcm).all())
(output / 'seam-metrics.json').write_text(json.dumps(report, indent=2)+'\n')
ffmpeg = os.environ['H3_FFMPEG']
for path in sorted(output.glob('*.mp4')):
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'], check=True)
(output / 'decode-check.txt').write_text('All MP4 files decoded without FFmpeg errors.\n')
