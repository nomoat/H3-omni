"""Fail closed unless all required phase-one reports and artifacts exist."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(root):
    contract_file = Path(__file__).with_name("acceptance.json")
    contract = json.loads(contract_file.read_text())
    spec = contract["required_end_to_end"]
    evidence = []

    def report(path):
        data = json.loads((root / path).read_text())
        if data.get("passed") is not True:
            raise ValueError(f"{path}: failed or missing pass status")
        evidence.append(str(path))
        return data

    if json.loads((root / "acceptance.json").read_text()) != contract:
        raise ValueError("Run used a different acceptance contract")
    if json.loads((root / "sources.json").read_text()) != json.loads(Path(__file__).with_name("sources.json").read_text()):
        raise ValueError("Run used different reference sources")
    primitives = report(Path("primitives.json"))
    if len(primitives.get("cases", [])) < 69 or not all(r["passed"] for r in primitives["cases"]):
        raise ValueError("Missing or failed primitive tests")
    if sum(r["name"].startswith("flash_") for r in primitives["cases"]) != 6:
        raise ValueError("Missing default FlashAttention comparisons")
    host = report(Path("host.json"))
    if len(host.get("cases", [])) != 12 or not all(r["sigma_exact"] and r["updates_exact"] for r in host["cases"]):
        raise ValueError("Production schedule or sampler differs")
    expected_layouts = {(tokens, width, height, frames)
                        for tokens in (6, 8, 11, 82)
                        for width, height in ((256, 256), (512, 512), (864, 480), (480, 864))
                        for frames in (22, 56)}
    layout_cases = host.get("layout_cases", [])
    actual_layouts = {(r["tokens"], r["width"], r["height"], r["frames"]) for r in layout_cases}
    if actual_layouts != expected_layouts or len(layout_cases) != len(expected_layouts) or not all(r["passed"] for r in layout_cases):
        raise ValueError("Production layout differs")
    text = report(Path("text/results.json"))
    if text.get("trace_enabled"):
        raise ValueError("Instrumented text runs cannot count as acceptance")
    ids = [r for r in text["cases"] if "native" in r]
    embeddings = [r for r in text["cases"] if "relative_l2" in r]
    if len(ids) < 9 or len(embeddings) < 2:
        raise ValueError("Missing tokenizer or real text encoder results")
    if not all(r["passed"] and r["native"] == r["reference"] for r in ids):
        raise ValueError("Tokenizer IDs differ")
    for row in embeddings:
        limits = contract["model"]["text_embedding"]
        if not row["passed"] or not (row["relative_l2"] <= limits["relative_l2"] and row["normalized_max"] <= limits["normalized_max"]):
            raise ValueError("Text embedding tolerance exceeded")
    for geometry in spec["cases"]:
        for seed in spec["seeds"]:
            name = f'{geometry["width"]}x{geometry["height"]}x{geometry["frames"]}-seed{seed}'
            folder = Path("trajectories") / name
            trajectory = report(folder / "results.json")
            arguments = trajectory["arguments"]
            if not trajectory["complete_trajectory"] or not trajectory["real_text"]:
                raise ValueError(name + ": diagnostic run cannot count as acceptance")
            if trajectory.get("diagnostic_attention") or trajectory.get("trace_enabled"):
                raise ValueError(name + ": diagnostic instrumentation cannot count as acceptance")
            if trajectory["layers"] != spec["layers"] or arguments["steps"] != spec["steps"]:
                raise ValueError(name + ": layer or step count differs")
            if trajectory.get("reference_timestep_plan") != "MiniMaxH3SetTimestepsStep.build_row_timesteps":
                raise ValueError(name + ": official reference timestep plan missing")
            if trajectory.get("reference_timestep_rows") != [1] + [2] * (spec["steps"] - 1):
                raise ValueError(name + ": official reference timestep row counts differ")
            if arguments["seed"] != seed or any(arguments[k] != v for k, v in geometry.items()):
                raise ValueError(name + ": seed or geometry differs")
            exact = trajectory.get("exact_checks", {})
            if any(exact.get(k) is not True for k in ("positions", "token_tags", "packing_indices", "latent_packing")):
                raise ValueError(name + ": exact layout checks missing")
            digest = hashlib.sha256((root / folder / "inputs.safetensors").read_bytes()).hexdigest()
            if digest != trajectory["input_sha256"]:
                raise ValueError(name + ": input fixture hash differs")
            rows = {r["label"]: r for r in trajectory["results"]}
            if len(rows) != 4 * spec["steps"]:
                raise ValueError(name + ": missing per-step comparisons")
            for step in range(spec["steps"]):
                for modality in ("video", "audio"):
                    for stage in ("velocity", "latent"):
                        row = rows[f"step_{step}_{modality}_{stage}"]
                        limits = contract["model"]["dit_velocity" if stage == "velocity" else "each_denoising_step"]
                        if not row["passed"] or not (row["relative_l2"] <= limits["relative_l2"] and row["normalized_max"] <= limits["normalized_max"]):
                            raise ValueError(name + ": " + row["label"] + " failed")
                if not (root / folder / f"step-{step:02d}.safetensors").is_file():
                    raise ValueError(name + ": missing velocity artifact")
            if not (root / folder / "latents.safetensors").is_file():
                raise ValueError(name + ": missing final latents")
            decoded = report(Path("decoded") / name / "results.json")
            rows = {(r["component"], r["scope"]): r for r in decoded["cases"]}
            for modality in ("video", "audio"):
                for scope in ("identical_latents", "independent_trajectory"):
                    row = rows[(modality, scope)]
                    limits = contract["model"][f"decoded_{modality}_float"]
                    if not row["passed"] or not (row["relative_l2"] <= limits["relative_l2"] and row["normalized_max"] <= limits["normalized_max"]):
                        raise ValueError(name + ": " + modality + " decode failed")
                    if not (root / "decoded" / name / f"{modality}-{scope}.safetensors").is_file():
                        raise ValueError(name + ": missing decoded tensor artifact")
    smoke = report(Path("cli-smoke.json"))
    if smoke.get("frames", 0) < 22 or smoke.get("steps") != spec["steps"] or smoke.get("layers") != spec["layers"]:
        raise ValueError("Missing full standalone CLI smoke test")
    if hashlib.sha256((root / "cli-smoke.mp4").read_bytes()).hexdigest() != smoke.get("movie_sha256"):
        raise ValueError("Standalone CLI media hash differs")
    if smoke.get("decoded_audio_bytes", 0) <= 0:
        raise ValueError("Standalone CLI produced no audio")
    return dict(passed=True, contract_sha256=hashlib.sha256(contract_file.read_bytes()).hexdigest(), evidence=evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.results)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(passed=False, error=str(error))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "acceptance-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
