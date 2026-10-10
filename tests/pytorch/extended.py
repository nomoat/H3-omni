"""Held-out long-prompt/default-geometry validation under the unchanged limits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec_path = Path(__file__).with_name("extended.json")
    spec = json.loads(spec_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "extended.json").write_bytes(spec_path.read_bytes())

    def run(script, *arguments):
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)),
                        "--weights", args.weights, *map(str, arguments)], check=True)

    run("text.py", "--prompt", spec["prompt"], "--output", args.output / "text")
    geometry = [value for name in ("width", "height", "frames", "steps", "seed")
                for value in ("--" + name, str(spec[name]))]
    run("real_dit.py", "--diffusers", args.diffusers,
        "--text", args.output / "text/text-2.safetensors", *geometry,
        "--output", args.output / "trajectory")
    run("decoders.py", "--diffusers", args.diffusers,
        "--latents", args.output / "trajectory/latents.safetensors",
        "--output", args.output / "decoded")
    reports = {name: json.loads((args.output / name / "results.json").read_text())
               for name in ("text", "trajectory", "decoded")}
    result = dict(passed=all(r["passed"] for r in reports.values()),
                  configuration=spec,
                  contract_sha256=hashlib.sha256(Path(__file__).with_name("acceptance.json").read_bytes()).hexdigest(),
                  reports={name: str(args.output / name / "results.json") for name in reports})
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
