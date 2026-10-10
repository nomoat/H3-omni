"""Locate the first hidden-state divergence in an instrumented DiT run."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    expected = load_file(str(args.directory / "trace-reference.safetensors"))
    rows = []
    for name in ["packed_input"] + [f"block-{i:02d}" for i in range(50)]:
        source = bytearray((args.directory / "trace-native" / f"{name}.bf16").read_bytes())
        reference = expected[name].float()
        native = torch.frombuffer(source, dtype=torch.bfloat16).reshape(reference.shape).float()
        delta = native - reference
        row = dict(stage=name, relative_l2=float(delta.norm() / reference.norm()),
                   normalized_max=float(delta.abs().max() / reference.abs().max()),
                   unequal_fraction=float((native != reference).float().mean()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.directory / "trace-summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
