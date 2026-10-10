"""Calibrate trajectory sensitivity: run the required case matrix without
stopping on failure and summarize how far native trajectories move from the
reference. Used to compare a candidate kernel (for example FlashAttention-3)
with an equally valid reference perturbation (PyTorch math attention).
Diagnostic only; it never counts as acceptance."""
import argparse
import gc
import json
from pathlib import Path

import torch
from real_dit import main as run_dit


def summarize(results_path):
    data = json.loads(Path(results_path).read_text())
    rows = [r for r in data["results"] if "label" in r]
    out = {}
    for kind in ("video_velocity", "audio_velocity", "video_latent", "audio_latent"):
        values = [(int(r["label"].split("_")[1]), r["relative_l2"], r["normalized_max"])
                  for r in rows if r["label"].endswith(kind)]
        if values:
            out[kind] = dict(step0_rel=values[0][1], step0_normmax=values[0][2], final_rel=values[-1][1],
                             max_rel=max(v[1] for v in values), max_normmax=max(v[2] for v in values))
    out["failed_labels"] = sum(1 for r in rows if not r["passed"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-math-diagnostic", action="store_true")
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name("acceptance.json").read_text())["required_end_to_end"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session, summary = {}, {}
    try:
        for geometry in spec["cases"]:
            for seed in spec["seeds"]:
                name = f'{geometry["width"]}x{geometry["height"]}x{geometry["frames"]}-seed{seed}'
                argv = ["--weights", args.weights, "--diffusers", args.diffusers, "--text", args.text,
                        "--width", str(geometry["width"]), "--height", str(geometry["height"]),
                        "--frames", str(geometry["frames"]), "--steps", str(spec["steps"]),
                        "--seed", str(seed), "--output", str(output / name)]
                if args.attention_math_diagnostic:
                    argv.append("--attention-math-diagnostic")
                report = run_dit(argv, cache=session)
                summary[name] = dict(passed=report["passed"], **summarize(output / name / "results.json"))
                (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
                print(json.dumps({name: summary[name]}), flush=True)
    finally:
        if "native" in session:
            session.pop("native").close()
        session.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
