"""Fetch the original, pinned FL2VA checkpoint; never quantize or prune it.

--ref2va also fetches the Ref2VA variant (144 GB), which ordered-reference
requests (--ref-image, --ref-video, ...) need. h3 reads its tokenizer, text
encoder, transformer and both VAEs from Ref2VA/. At revision 42ed227e only the
transformer differs from FL2VA; the other files are byte-identical."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("destination")
p.add_argument("--ref2va", action="store_true", help="also fetch the Ref2VA variant")
args = p.parse_args()
spec = json.loads(Path(__file__).with_name("sources.json").read_text())["checkpoint"]
snapshot_download(
    spec["repository"], revision=spec["revision"], local_dir=args.destination,
    allow_patterns=["model_index.json", "LICENSE*", "FL2VA/*"]
    + (["Ref2VA/*"] if args.ref2va else []), max_workers=8,
)
print("Checkpoint download complete", flush=True)
