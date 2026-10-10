"""Fetch the original, pinned FL2VA checkpoint; never quantize or prune it."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("destination")
args = p.parse_args()
spec = json.loads(Path(__file__).with_name("sources.json").read_text())["checkpoint"]
snapshot_download(
    spec["repository"], revision=spec["revision"], local_dir=args.destination,
    allow_patterns=["model_index.json", "LICENSE*", "FL2VA/*"], max_workers=8,
)
print("Checkpoint download complete", flush=True)
