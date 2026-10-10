"""Histogram of GPU idle gaps in one phase of one request.

Usage: gap_histogram.py TRACE EVENTS REQUEST PHASE NEXT_PHASE [TOP]
e.g. gap_histogram.py trace.txt.gz serve/events.jsonl p10 "video VAE load" FFmpeg 20
Prints idle time by gap length and the TOP longest gaps with the GPU
operations before and after each one (M = memcpy "bytes kind", K = kernel).
"""
import gzip, json, sys
trace, events, req, phase, nxt = sys.argv[1:6]
ev = next(e for e in map(json.loads, open(events)) if e.get('event') == 'request' and e['name'] == req)
ph = {p['phase']: p for p in ev['phases']}
lo = int((ev['start'] + ph[phase]['first']) * 1e9)
hi = int((ev['start'] + ph[nxt]['first']) * 1e9)
off = 0
gpu = []
for line in gzip.open(trace, 'rt') if trace.endswith('.gz') else open(trace):
    k = line[0]
    if k == 'C':
        _, c, m = line.split()
        off = int(m) - int(c)
    elif k in 'KMS':
        p = line.split()
        s, e = int(p[1]) + off, int(p[2]) + off
        if lo <= s < hi:
            gpu.append((s, e, k, p[-1] if k == 'K' else ' '.join(p[3:])))
gpu.sort()
cur = lo
gaps = []
prev = None
for s, e, k, n in gpu:
    if s > cur:
        gaps.append((s - cur, prev, (k, n[:50]), cur))
    if e > cur:
        cur = e
        prev = (k, n[:50])
gaps.append((hi - cur, prev, None, cur))
bins = [(0, 5e3), (5e3, 2e4), (2e4, 1e5), (1e5, 1e6), (1e6, 1e7), (1e7, 1e8), (1e8, 1e11)]
for a, b in bins:
    sel = [g for g in gaps if a <= g[0] < b]
    print(f"gap {a/1e3:8.0f}-{b/1e3:9.0f} us: n={len(sel):7d} total={sum(g[0] for g in sel)/1e9:7.3f} s")
print('window', (hi - lo) / 1e9, 'idle', sum(g[0] for g in gaps) / 1e9)
big = sorted(gaps, key=lambda g: -g[0])[:int(sys.argv[6]) if len(sys.argv) > 6 else 40]
for g in sorted(big, key=lambda g: g[3]):
    print(f"{(g[3]-lo)/1e9:8.3f} s  gap {g[0]/1e6:8.2f} ms  after {g[1]}  before {g[2]}")
