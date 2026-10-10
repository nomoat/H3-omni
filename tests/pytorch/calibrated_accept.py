"""Calibrated acceptance for kernels that are as accurate as the reference
but round differently (FlashAttention-3 instead of the PyTorch
FlashAttention-2 build).

The 20-step sampler amplifies any rounding change: PyTorch's own math SDPA,
which is more accurate than its default kernel, moves the final latents by
4-31% (tests/pytorch/evidence/calibration/math-sdpa-14433.json). The
independent-trajectory limits in acceptance.json therefore only pass for the
identical kernel. This check keeps the single-forward limits and bounds the
trajectory drift by the reference's own math-SDPA drift.

Limits, derived from the math-SDPA baseline before the candidate run:
  * every case completes with finite values;
  * step-0 video/audio velocity (identical inputs, one forward) within the
    acceptance.json dit_velocity limits;
  * mean and maximum final video/audio latent drift over the six cases at
    most 1.5x the math-SDPA mean and maximum.
"""
import argparse
import json
import math
from pathlib import Path

FACTOR = 1.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="calibrate.py summary.json")
    parser.add_argument("--baseline", default=str(Path(__file__).parent / "evidence/calibration/math-sdpa-14433.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    limits = json.loads(Path(__file__).with_name("acceptance.json").read_text())["model"]["dit_velocity"]
    spec = json.loads(Path(__file__).with_name("acceptance.json").read_text())["required_end_to_end"]
    cases = [f'{g["width"]}x{g["height"]}x{g["frames"]}-seed{s}' for g in spec["cases"] for s in spec["seeds"]]
    candidate = json.loads(Path(args.candidate).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    checks = []

    def check(name, value, limit):
        ok = math.isfinite(value) and value <= limit
        checks.append(dict(check=name, value=value, limit=limit, passed=ok))

    missing = [c for c in cases if c not in candidate]
    checks.append(dict(check="all cases present", missing=missing, passed=not missing))
    for case in cases:
        if case in missing:
            continue
        for kind in ("video_velocity", "audio_velocity"):
            check(f"{case} {kind} step0 relative_l2", candidate[case][kind]["step0_rel"], limits["relative_l2"])
            if "step0_normmax" in candidate[case][kind]:
                check(f"{case} {kind} step0 normalized_max", candidate[case][kind]["step0_normmax"], limits["normalized_max"])
    for kind in ("video_latent", "audio_latent"):
        got = [candidate[c][kind]["final_rel"] for c in cases if c not in missing]
        ref = [baseline[c][kind]["final_rel"] for c in cases]
        if got:
            check(f"{kind} final drift mean", sum(got) / len(got), FACTOR * sum(ref) / len(ref))
            check(f"{kind} final drift max", max(got), FACTOR * max(ref))
    report = dict(passed=all(c["passed"] for c in checks), factor=FACTOR, checks=checks)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    for c in checks:
        if not c["passed"] or "drift" in c["check"]:
            print(json.dumps(c))
    print(json.dumps(dict(passed=report["passed"])))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
