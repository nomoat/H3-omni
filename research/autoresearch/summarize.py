"""Summarize the exploratory native probe; these are not profiler timings."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
media = json.loads((root / "media-check.json").read_text())
results = []
for result in (event for event in events if event["event"] == "result"):
    request = result["request"]
    progress = [e for e in events if e.get("request") == request and e["event"] == "progress"]
    first = {}
    for event in progress:
        first.setdefault(event["phase"], event["seconds"])
    boundaries = [
        ("request_setup_text_and_dit_preparation", 0.0),
        ("denoising_including_host_work_and_transfers", first["denoise"]),
        ("audio_vae_load_decode_and_cleanup", first["audio VAE"]),
        ("video_vae_cache_or_load_decode_and_rgb_conversion", first["video VAE load"]),
        ("media_encoding_and_return", first["FFmpeg"]),
        ("end", result["seconds"]),
    ]
    spans = {name: boundaries[i + 1][1] - start
             for i, (name, start) in enumerate(boundaries[:-1])}
    assert all(value >= 0 for value in spans.values())
    assert abs(sum(spans.values()) - result["seconds"]) < 1e-6
    results.append({**result, "coarse_host_wall_spans_seconds": spans})

summary = {
    "job": 14319,
    "status": "COMPLETED",
    "seed_release": 14274,
    "gpu": "1 NVIDIA H200",
    "cpu_allocation": 4,
    "context_load_seconds": next(e["seconds"] for e in events if e["event"] == "context_load"),
    "timing_boundary": "From immediately before h3_generate to its return after the output file is closed; context loading is separate.",
    "cache": "h3_cache_set_enabled(ctx, 1); all three requests use one process in the listed order.",
    "samples_per_mode": 1,
    "results": results,
    "media_check": media,
    "limitations": [
        "Exploratory observations, not a statistically confirmed benchmark or an optimization speedup.",
        "Phase spans come from CPU progress callbacks, include setup/transfers/cleanup, and are not kernel profiler measurements.",
        "The node was shared with other jobs; the allocated GPU was exclusive, but shared host/storage contention was not controlled.",
        "First-request timing uses available local/filesystem caches and does not include process launch or context loading; it is not a cold-disk measurement.",
        "Mode, seed, prompt and request order vary; the measurements do not isolate the causal cost of any one cache.",
        "This probe links unchanged accepted inference objects. It checks media completeness, not a new full numerical acceptance run for all three inputs.",
        "The future frozen evaluator must separately implement resident fresh-input timing and independent numerical comparison; this probe is not that evaluator."
    ],
    "evidence_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("events.jsonl", "media-check.json", "source-and-object-sha256.txt", "baseline.c", "prompt-a.txt", "prompt-b.txt")
    }
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
