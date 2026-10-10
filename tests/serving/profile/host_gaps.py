"""Attribute GPU idle time to host CUDA API calls (trace with H3_CUPTI_API=1).

For one phase of one request, find every interval where no kernel, copy or
memset runs, then charge each idle interval to the host API calls that
overlap it. Idle time with no overlapping API call is host-side C work.
"""
import argparse
import collections
import gzip
import json


def open_text(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--phase', default='denoise')
    parser.add_argument('--next-phase', default='audio VAE')
    args = parser.parse_args()
    event = next(e for e in map(json.loads, open(args.events))
                 if e.get('event') == 'request' and e['name'] == args.request)
    phases = {p['phase']: p for p in event['phases']}
    lo = int((event['start'] + phases[args.phase]['first']) * 1e9)
    hi = int((event['start'] + phases[args.next_phase]['first']) * 1e9)
    offset, gpu, api = 0, [], []
    for line in open_text(args.trace):
        kind = line[0]
        if kind == 'C':
            _, cupti, mono = line.split()
            offset = int(mono) - int(cupti)
        elif kind in 'KMS':
            parts = line.split()
            start, end = int(parts[1]) + offset, int(parts[2]) + offset
            if lo <= start < hi:
                gpu.append((start, end))
        elif kind == 'A':
            parts = line.split()
            start, end = int(parts[1]) + offset, int(parts[2]) + offset
            if end > lo and start < hi:
                api.append((start, end, parts[3] + ':' + parts[6]))
    gpu.sort()
    idle, cursor = [], lo
    for start, end in gpu:
        if start > cursor:
            idle.append((cursor, start))
        cursor = max(cursor, end)
    if hi > cursor:
        idle.append((cursor, hi))
    api.sort()
    charged = collections.Counter()
    totals = collections.Counter()
    counts = collections.Counter()
    for start, end, name in api:
        totals[name] += min(end, hi) - max(start, lo)
        counts[name] += 1
    uncovered = 0
    index = 0
    for start, end in idle:
        # API calls overlapping [start, end); charge the overlap.
        while index < len(api) and api[index][1] <= start - 10_000_000:
            index += 1
        covered = []
        for a_start, a_end, name in api[index:]:
            if a_start >= end:
                break
            overlap = min(end, a_end) - max(start, a_start)
            if overlap > 0:
                charged[name] += overlap
                if name.startswith('R:'):
                    covered.append((max(start, a_start), min(end, a_end)))
        covered.sort()
        run, edge = 0, start
        for c_start, c_end in covered:
            c_start = max(c_start, edge)
            if c_end > c_start:
                run += c_end - c_start
                edge = c_end
        uncovered += (end - start) - run
    window = hi - lo
    total_idle = sum(e - s for s, e in idle)
    print(f'{args.request} {args.phase}: window {window/1e9:.2f} s, '
          f'GPU idle {total_idle/1e9:.2f} s in {len(idle)} gaps, '
          f'idle not inside any runtime call {uncovered/1e9:.2f} s')
    print('API time overlapping GPU idle (s) | all time in API (s) | calls')
    for name, value in charged.most_common(20):
        print(f'  {value/1e9:8.3f} | {totals[name]/1e9:8.3f} | {counts[name]:7d}  {name}')


if __name__ == '__main__':
    main()
