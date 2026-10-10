"""Summarize a CUPTI trace (cupti_trace.c) per request phase.

Phase windows come from h3_serve_check events: each phase runs from its first
progress callback to the next phase's first callback; the last runs to the
request end. For each window the report gives wall time, GPU busy time (union
of kernel/copy intervals), time by kernel category and the top kernels.
"""
import argparse
import collections
import json
import re

CATEGORIES = [
    ('attention', re.compile(r'flash|fmha|attention', re.I)),
    ('gemm', re.compile(r'gemm|xmma|nvjet|cutlass|sm90_|sm80_|cublas', re.I)),
    ('conv', re.compile(r'conv|cudnn|implicit|winograd|fft', re.I)),
    ('h3', re.compile(r'h3_')),
]


def category(name):
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return 'other'


def short(name, width=110):
    return name if len(name) <= width else name[:width - 3] + '...'


def load_trace(path):
    offset = None
    records = []
    with open(path) as source:
        for line in source:
            kind = line[0]
            if kind == 'C':
                _, cupti, mono = line.split()
                offset = int(mono) - int(cupti)
            elif kind == 'K':
                parts = line.rstrip('\n').split(' ', 10)
                records.append(('K', int(parts[1]), int(parts[2]), parts[10],
                                tuple(int(v) for v in parts[4:10])))
            elif kind in 'MS':
                parts = line.split()
                label = 'memcpy' if kind == 'M' else 'memset'
                if kind == 'M':
                    label += {'1': ' H2D', '2': ' D2H', '8': ' D2D'}.get(parts[4], ' ' + parts[4])
                records.append((kind, int(parts[1]), int(parts[2]), label,
                                int(parts[3])))
    records.sort(key=lambda r: r[1])
    return offset, records


def union(intervals):
    total, end = 0, None
    for start, stop in intervals:
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(records, lo, hi, top):
    window = [r for r in records if r[1] >= lo and r[1] < hi]
    by_cat = collections.Counter()
    by_name = collections.defaultdict(lambda: [0, 0])
    for kind, start, stop, name, _ in window:
        duration = stop - start
        cat = category(name) if kind == 'K' else name.split()[0]
        by_cat[cat] += duration
        entry = by_name[(cat, name)]
        entry[0] += duration
        entry[1] += 1
    wall = hi - lo
    busy = union((r[1], r[2]) for r in window)
    kernels = sum(1 for r in window if r[0] == 'K')
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]
    return dict(
        wall_s=wall / 1e9, busy_s=busy / 1e9,
        busy_fraction=busy / wall if wall else 0.0,
        kernels=kernels,
        mean_kernel_us=(sum(r[2] - r[1] for r in window if r[0] == 'K') /
                        kernels / 1e3) if kernels else 0.0,
        categories={k: v / 1e9 for k, v in by_cat.most_common()},
        top=[dict(category=cat, name=short(name), seconds=v[0] / 1e9,
                  count=v[1], mean_us=v[0] / v[1] / 1e3)
             for (cat, name), v in ranked])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--requests', required=True,
                        help='comma list of request names to report')
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    offset, records = load_trace(args.trace)
    if offset is None:
        raise SystemExit('trace has no clock line')
    # Convert GPU timestamps to CLOCK_MONOTONIC nanoseconds.
    records = [(k, s + offset, e + offset, n, x) for k, s, e, n, x in records]
    wanted = args.requests.split(',')
    report = {}
    with open(args.events) as source:
        for line in source:
            event = json.loads(line)
            if event.get('event') != 'request' or event['name'] not in wanted:
                continue
            base = event['start']
            end = base + event['seconds']
            phases = sorted(event['phases'], key=lambda p: p['first'])
            windows = []
            for index, phase in enumerate(phases):
                lo = base + phase['first']
                hi = base + phases[index + 1]['first'] if index + 1 < len(phases) else end
                windows.append((phase['phase'].replace('video VAE load', 'video VAE'), lo, hi))
            ns = lambda t: int(round(t * 1e9))
            report[event['name']] = dict(
                seconds=event['seconds'],
                whole=summarize(records, ns(base), ns(end), args.top),
                phases={name: summarize(records, ns(lo), ns(hi), args.top)
                        for name, lo, hi in windows if hi - lo > 0.05})
    with open(args.output, 'w') as sink:
        json.dump(report, sink, indent=1)
    for name, entry in report.items():
        print(f'== {name}: {entry["seconds"]:.1f} s')
        for phase, data in entry['phases'].items():
            cats = ', '.join(f'{k} {v:.2f}' for k, v in data['categories'].items())
            print(f'  {phase:16s} wall {data["wall_s"]:7.2f}  busy {data["busy_s"]:7.2f} '
                  f'({data["busy_fraction"]:.0%})  kernels {data["kernels"]:7d} '
                  f'mean {data["mean_kernel_us"]:7.1f} us | {cats}')
            for item in data['top'][:12]:
                print(f'      {item["seconds"]:7.2f} s {item["count"]:7d}x '
                      f'{item["mean_us"]:9.1f} us  [{item["category"]}] {item["name"]}')


if __name__ == '__main__':
    main()
