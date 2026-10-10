"""Decode every required trajectory; missing inputs are errors."""
import argparse
import json
from pathlib import Path

from decoders import main as run_decoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name("acceptance.json").read_text())["required_end_to_end"]
    for geometry in spec["cases"]:
        for seed in spec["seeds"]:
            name = f'{geometry["width"]}x{geometry["height"]}x{geometry["frames"]}-seed{seed}'
            report = run_decoders([
                "--weights", args.weights, "--diffusers", args.diffusers,
                "--latents", str(Path(args.trajectories) / name / "latents.safetensors"),
                "--output", str(Path(args.output) / name),
            ])
            if not report["passed"]:
                raise RuntimeError("Decoder acceptance failed: " + name)


if __name__ == "__main__":
    main()
