"""Required geometry/seed matrix; reuse loaded weights without reusing predictions."""
import argparse
import gc
import json
from pathlib import Path

import torch
from real_dit import main as run_dit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name("acceptance.json").read_text())["required_end_to_end"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session, results = {}, []
    try:
        for geometry in spec["cases"]:
            for seed in spec["seeds"]:
                name = f'{geometry["width"]}x{geometry["height"]}x{geometry["frames"]}-seed{seed}'
                report = run_dit([
                    "--weights", args.weights, "--diffusers", args.diffusers, "--text", args.text,
                    "--width", str(geometry["width"]), "--height", str(geometry["height"]),
                    "--frames", str(geometry["frames"]), "--steps", str(spec["steps"]),
                    "--seed", str(seed), "--output", str(output / name),
                ], cache=session)
                results.append({"case": name, "passed": report["passed"]})
                (output / "suite-progress.json").write_text(json.dumps(results, indent=2) + "\n")
                if not report["passed"]:
                    raise RuntimeError("Trajectory acceptance failed: " + name)
    finally:
        if "native" in session:
            session.pop("native").close()
        session.clear()
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"passed": True, "cases": results}), flush=True)


if __name__ == "__main__":
    main()
