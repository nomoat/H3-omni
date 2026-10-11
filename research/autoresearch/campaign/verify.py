#!/usr/bin/env python3
"""Frozen correctness judge for the H3 CUDA optimization campaign.

prepare creates a hash-addressed CPU oracle from accepted job 14274 artifacts.
screen runs the fixed 864x480x56 case, including its independent 20-step state.
promote runs all six regression cases plus the extended case against the cache.

This is an evaluation tool, not a performance timer. A screening PASS is not a
promotion PASS. The phase-one bridge covers the current campaign's CUDA backend
changes; changes to the production orchestration/device sampler need a new,
independently reviewed production-path numerical check before promotion.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests/pytorch"
CONTRACT_SHA256 = "fcece9d35dacbbf79dac53838bb0df625a1c457319dc250cb326de579fde2f72"
SOURCES_SHA256 = "2cb57568b4ad139ef5005b82df84c2899dc2fdfd598c941a558a624d48989adc"
SCHEMA = "h3-autoresearch-oracle-v2"
TIMESTEP_PLAN = "MiniMaxH3SetTimestepsStep.build_row_timesteps"
sys.path.insert(0, str(TESTS))


class NumericMismatch(ValueError):
    """An observed candidate value or output shape differs from the oracle."""


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_contract():
    require(digest(TESTS / "acceptance.json") == CONTRACT_SHA256, "Acceptance contract changed")
    require(digest(TESTS / "sources.json") == SOURCES_SHA256, "Reference source pins changed")
    return read_json(TESTS / "acceptance.json")


def harness_hashes():
    paths = list(TESTS.glob("*.py")) + [TESTS / "acceptance.json", TESTS / "sources.json",
            TESTS / "extended.json", Path(__file__), Path(__file__).with_name("reference_checkout.json"),
            ROOT / "h3_gpu.h", ROOT / "cuda/h3_cuda.h"]
    unit_judge = Path(__file__).with_name("frozen_primitives.py")
    if unit_judge.exists():
        paths.append(unit_judge)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def dependencies():
    import torch
    from safetensors.torch import load_file, save_file
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents,
    )
    return torch, load_file, save_file, MiniMaxH3Scheduler, MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents


def check_reference_checkout(checkout):
    checkout = Path(checkout).resolve()
    wanted = read_json(TESTS / "sources.json")["pytorch_reference"]["revision"]
    manifest = read_json(Path(__file__).with_name("reference_checkout.json"))
    require(manifest["revision"] == wanted, "Diffusers content manifest revision differs")
    require("src/diffusers/__init__.py" in manifest["files"] and
            "scripts/convert_minimax_h3_to_diffusers.py" in manifest["files"],
            "Diffusers content manifest lacks required reference sources")
    for relative, expected in manifest["files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Invalid Diffusers content manifest path")
        require(digest(checkout / relative) == expected, "Pinned Diffusers file content changed: " + relative)
    expected_python = {name for name in manifest["files"] if name.startswith("src/diffusers/") and name.endswith(".py")}
    actual_python = {str(path.relative_to(checkout)) for path in (checkout / "src/diffusers").rglob("*.py")}
    require(actual_python == expected_python, "Pinned Diffusers Python source file set changed")
    import diffusers
    require(Path(diffusers.__file__).resolve().is_relative_to(checkout / "src"),
            "Imported Diffusers must come from the pinned checkout/src")


def unpack_video(rows, shape):
    _, channels, frames, height, width = shape
    return rows.reshape(1, frames, height // 2, width // 2, channels, 1, 2, 2).permute(
        0, 4, 1, 5, 2, 6, 3, 7).reshape(shape).contiguous()


def unpack_audio(rows, shape):
    return rows.reshape(1, shape[2], shape[3], shape[1]).permute(0, 3, 1, 2).contiguous()


def same_bits(left, right):
    import torch
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def check_accepted_trajectory(report, spec):
    require(report.get("passed") is True and report.get("complete_trajectory") is True,
            "Incomplete or failed archived trajectory")
    require(report.get("real_text") is True and report.get("layers") == 50,
            "Archived trajectory lacks real text or all 50 layers")
    require(not report.get("trace_enabled") and not report.get("diagnostic_attention"),
            "Instrumented/diagnostic trajectory cannot create oracle")
    require(report.get("reference_timestep_plan") == TIMESTEP_PLAN and
            report.get("reference_timestep_rows") == [1] + [2] * 19,
            "Archived oracle used a different timestep plan")
    for key in ("width", "height", "frames", "steps", "seed"):
        require(report["arguments"][key] == spec[key], "Archived geometry/seed differs: " + key)
    require(all(report.get("exact_checks", {}).get(key) is True for key in
                ("positions", "token_tags", "packing_indices", "latent_packing")),
            "Archived exact layout checks missing")
    expected = {f"step_{step}_{modality}_{kind}" for step in range(20)
                for modality in ("video", "audio") for kind in ("velocity", "latent")}
    rows = report["results"]
    require(len(rows) == len(expected) and {r["label"] for r in rows} == expected,
            "Archived trajectory is missing comparisons")
    require(all(r["passed"] is True and r["relative_l2"] == 0 and r["normalized_max"] == 0 for r in rows),
            "Only the exactly matched 14274 trajectory can be converted to this cached oracle")


def case_specs():
    contract = frozen_contract()
    english = "A cat walks through a sunny garden."
    result = []
    for geometry in contract["required_end_to_end"]["cases"]:
        for seed in contract["required_end_to_end"]["seeds"]:
            key = f'{geometry["width"]}x{geometry["height"]}x{geometry["frames"]}-seed{seed}'
            result.append({"key": key, "configuration": {**geometry, "seed": seed, "steps": 20, "prompt": english},
                "trajectory": "trajectories/" + key, "decoded": "decoded/" + key,
                "text": "text", "text_index": 0, "text_key": "english"})
    result.append({"key": "extended", "configuration": read_json(TESTS / "extended.json"),
                   "trajectory": "extended/trajectory", "decoded": "extended/decoded",
                   "text": "extended/text", "text_index": 2, "text_key": "long_prompt"})
    return result


def validate_units(root):
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == 1 and
            tuple(len(manifest[key]) for key in ("primitive_cases", "host_cases", "layout_cases")) == (69, 12, 32),
            "Incomplete or unsupported frozen unit oracle")
    require(manifest["limits"] == frozen_contract()["primitives"], "Unit oracle changed precision limits")
    provenance = manifest["provenance"]
    require(provenance["capture_source_sha256"] == digest(Path(__file__).with_name("frozen_primitives.py")),
            "Unit oracle capture implementation changed")
    expected = {name: digest(TESTS / name) for name in ("primitives.py", "host.py", "native.py", "acceptance.json", "sources.json")}
    require(provenance["source_sha256"] == expected, "Unit oracle input generator or precision contract changed")
    for name, wanted in manifest["files"].items():
        require((root / name).resolve().parent == root.resolve() and digest(root / name) == wanted,
                "Frozen unit artifact changed: " + name)
    return manifest


def prepare_case(args, case, output):
    contract = frozen_contract()
    torch, load_file, save_file, Scheduler, Layout, Times, patchify = dependencies()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    source = args.reference_root.resolve()
    spec = case["configuration"]
    output.mkdir()
    archived = read_json(source / case["trajectory"] / "results.json")
    check_accepted_trajectory(archived, spec)
    provenance = {}

    def remember(relative):
        path = source / relative
        provenance[relative] = digest(path)
        return path

    for relative in (case["trajectory"] + "/results.json", case["text"] + "/results.json", case["decoded"] + "/results.json"):
        remember(relative)
    inputs_path = remember(case["trajectory"] + "/inputs.safetensors")
    require(digest(inputs_path) == archived["input_sha256"], "Archived input fixture hash differs")
    inputs = load_file(str(inputs_path))
    require(same_bits(inputs["text"], inputs["reference_text"]), "Archived text was not exact")
    text = load_file(str(remember(f'{case["text"]}/text-{case["text_index"]}.safetensors')))
    require(same_bits(text["text"], inputs["text"]) and same_bits(text["reference_text"], inputs["reference_text"]),
            "Text fixture does not match archived trajectory")
    text_report = read_json(source / case["text"] / "results.json")
    ids = [r for r in text_report["cases"] if "native" in r and r["prompt"] == spec["prompt"]]
    require(text_report.get("passed") is True and not text_report.get("trace_enabled") and len(ids) == 1,
            "Missing accepted long-prompt tokenizer result")
    require(ids[0]["passed"] is True and ids[0]["native"] == ids[0]["reference"], "Archived token IDs differ")
    require(inputs["text"].shape == (1, len(ids[0]["reference"]), 5120), "Archived embedding shape differs")
    video, audio = inputs["video"].clone(), inputs["audio"].clone()
    require(video.dtype == audio.dtype == torch.float32 and inputs["text"].dtype == torch.bfloat16,
            "Oracle dtypes differ")
    schedulers = [Scheduler(shift=shift) for shift in (12, 3)]
    for scheduler in schedulers:
        scheduler.set_timesteps(21)
    positions, tags, vi, ai, ti = Layout.build_packed_sequence(
        torch.ones(inputs["text"].shape[1], dtype=torch.long), *video.shape[2:],
        audio.shape[-1], (1, 2, 2), 2, 2, 0)[:5]
    counts = [Times.build_row_timesteps(vi, ai, 0, 0, ti.numel(), float(vt), float(at),
               max(float(vt), .999), 1.0)[0].numel()
              for vt, at in zip(schedulers[0].timesteps, schedulers[1].timesteps)]
    require(counts == [1] + [2] * 19, "Pinned helper timestep rows changed")
    initial = {"text": inputs["reference_text"], "video": video.clone(), "audio": audio.clone(),
               "positions": positions, "tags": tags, "video_indices": vi, "audio_indices": ai,
               "text_indices": ti, "token_ids": torch.tensor(ids[0]["reference"], dtype=torch.int32),
               "video_sigmas": schedulers[0].sigmas, "audio_sigmas": schedulers[1].sigmas}
    save_file({k: v.contiguous() for k, v in initial.items()}, str(output / "initial.safetensors"))
    for step in range(20):
        archived_step = load_file(str(remember(f'{case["trajectory"]}/step-{step:02d}.safetensors')))
        pv, pa = archived_step["reference_video_packed"], archived_step["reference_audio_packed"]
        require(same_bits(patchify(archived_step["video_velocity"], (1, 2, 2)).unsqueeze(0), pv),
                f"Archived native/reference video velocity not exact at step {step}")
        require(same_bits(archived_step["audio_velocity"].permute(0, 2, 3, 1).reshape(1, -1, 32), pa),
                f"Archived native/reference audio velocity not exact at step {step}")
        item = {"video_before": video, "audio_before": audio,
                "video_velocity_packed": pv, "audio_velocity_packed": pa}
        video = schedulers[0].step(unpack_video(pv, video.shape), schedulers[0].timesteps[step], video).prev_sample
        audio = schedulers[1].step(unpack_audio(pa, audio.shape), schedulers[1].timesteps[step], audio).prev_sample
        item.update(video_after=video, audio_after=audio)
        require(all(bool(torch.isfinite(v).all()) for v in item.values()), "Non-finite archived oracle")
        save_file({k: v.contiguous() for k, v in item.items()}, str(output / f"step-{step:02d}.safetensors"))
    final = load_file(str(remember(case["trajectory"] + "/latents.safetensors")))
    require(same_bits(video, final["reference_video"]) and same_bits(audio, final["reference_audio"]),
            "CPU official-scheduler reconstruction does not exactly match archived final reference state")
    require(same_bits(video, final["video"]) and same_bits(audio, final["audio"]),
            "Archived native final state was not exact")
    save_file({"video": video, "audio": audio}, str(output / "final.safetensors"))
    decoded_report = read_json(source / case["decoded"] / "results.json")
    require(decoded_report.get("passed") is True, "Archived decoder failed")
    for modality in ("video", "audio"):
        values = {}
        for scope in ("identical_latents", "independent_trajectory"):
            rows = [r for r in decoded_report["cases"] if r["component"] == modality and r["scope"] == scope]
            require(len(rows) == 1 and rows[0]["passed"] is True, "Missing archived decoded comparison")
            limits = contract["model"][f"decoded_{modality}_float"]
            require(all(math.isfinite(rows[0][k]) and rows[0][k] <= limits[k]
                        for k in ("relative_l2", "normalized_max")), "Archived decoder exceeded fixed limits")
            tensors = load_file(str(remember(f'{case["decoded"]}/{modality}-{scope}.safetensors')))
            expected = tensors["reference"]
            require(expected.dtype == torch.float32 and bool(torch.isfinite(expected).all()), "Invalid decoded oracle")
            values[scope] = expected.contiguous()
        save_file(values, str(output / f"{modality}-decoded.safetensors"))
    return {**case, "provenance_sha256": provenance,
            "files": {p.name: digest(p) for p in sorted(output.glob("*.safetensors"))}}


def prepare(args, report):
    contract = frozen_contract()
    check_reference_checkout(args.diffusers)
    source, output = args.reference_root.resolve(), args.output.resolve()
    require(read_json(source / "acceptance.json") == contract, "Archive acceptance contract differs")
    require(read_json(source / "sources.json") == read_json(TESTS / "sources.json"), "Archive source pins differ")
    require(read_json(source / "extended/extended.json") == read_json(TESTS / "extended.json"), "Archived extended fixture differs")
    cases = [prepare_case(args, case, output / case["key"]) for case in case_specs()]
    torch, load_file, save_file, *_ = dependencies()
    texts = []
    for key, prefix, index, prompt in (("english", "text", 0, "A cat walks through a sunny garden."),
             ("chinese", "text", 1, "\u4e00\u53ea\u732b\u5728\u82b1\u56ed\u91cc\u8d70\u52a8\u3002"),
             ("long_prompt", "extended/text", 2, read_json(TESTS / "extended.json")["prompt"])):
        archive = source / prefix / f"text-{index}.safetensors"
        tensors = load_file(str(archive))
        require(same_bits(tensors["text"], tensors["reference_text"]), "Archived text cache must be exact")
        results = read_json(source / prefix / "results.json")
        ids = [r for r in results["cases"] if r["prompt"] == prompt and "native" in r]
        embeddings = [r for r in results["cases"] if r["prompt"] == prompt and "relative_l2" in r]
        require(results.get("passed") is True and not results.get("trace_enabled") and len(ids) == len(embeddings) == 1,
                "Missing accepted text oracle")
        require(ids[0]["native"] == ids[0]["reference"] and ids[0]["passed"] is True and
                embeddings[0]["passed"] is True and embeddings[0]["relative_l2"] == embeddings[0]["normalized_max"] == 0,
                "Archived text was not exact")
        path = f"text-{key}.safetensors"
        save_file({"text": tensors["reference_text"], "token_ids": torch.tensor(ids[0]["reference"], dtype=torch.int32)}, str(output / path))
        texts.append({"key": key, "prompt": prompt, "file": path,
                      "source_sha256": digest(archive), "source_report_sha256": digest(source / prefix / "results.json")})
    token_cases = {}
    for prefix in ("text", "extended/text"):
        for row in read_json(source / prefix / "results.json")["cases"]:
            if "native" not in row:
                continue
            require(row["passed"] is True and row["native"] == row["reference"], "Archived tokenizer case failed")
            require(row["prompt"] not in token_cases or token_cases[row["prompt"]] == row["reference"], "Archived tokenizer references disagree")
            token_cases[row["prompt"]] = row["reference"]
    require(len(token_cases) == 10, "Expected nine tokenizer regressions and the extended prompt")
    write_json(output / "tokenizer.json", {"cases": [{"prompt": k, "reference": v} for k, v in token_cases.items()]})
    units = None
    if args.unit_oracle is not None:
        require((args.unit_oracle / "manifest.json").is_file(), "Unit oracle manifest missing")
        validate_units(args.unit_oracle)
        shutil.copytree(args.unit_oracle, output / "units")
        units = {"directory": "units", "manifest_sha256": digest(output / "units/manifest.json")}
    manifest = {"schema": SCHEMA, "source_job": 14274, "configuration": read_json(TESTS / "extended.json"),
                "contract_sha256": CONTRACT_SHA256, "sources_sha256": SOURCES_SHA256,
                "harness_sha256": harness_hashes(), "cases": cases, "texts": texts, "units": units,
                "reference_timestep_plan": TIMESTEP_PLAN, "reference_timestep_rows": [1] + [2] * 19,
                "reference_reconstruction": "Archived reference velocities; pinned CPU scheduler; final state checked bitwise against archived independent CUDA reference",
                "velocity_contract": "Candidate velocity on fixed reference inputs; independent candidate latents compared after every step. No candidate-state live PyTorch oracle is run.",
                "files": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    report.update(manifest_sha256=digest(output / "manifest.json"), oracle=str(output), cases=7, text_embeddings=3,
                  ready_for_campaign=units is not None)


def load_manifest(args):
    frozen_contract()
    require(digest(args.oracle / "manifest.json") == args.oracle_sha256, "Frozen oracle manifest hash differs")
    manifest = read_json(args.oracle / "manifest.json")
    require(manifest["schema"] == SCHEMA and manifest["source_job"] == 14274, "Unsupported oracle provenance")
    require(manifest["contract_sha256"] == CONTRACT_SHA256 and manifest["sources_sha256"] == SOURCES_SHA256,
            "Oracle precision/source pins differ")
    require(manifest["configuration"] == read_json(TESTS / "extended.json"), "Oracle fixed input differs")
    require(manifest["harness_sha256"] == harness_hashes(), "Frozen harness changed since oracle preparation")
    require(manifest["reference_timestep_plan"] == TIMESTEP_PLAN and manifest["reference_timestep_rows"] == [1] + [2] * 19,
            "Oracle timestep metadata differs")
    expected_texts = [("english", "A cat walks through a sunny garden."), ("chinese", "\u4e00\u53ea\u732b\u5728\u82b1\u56ed\u91cc\u8d70\u52a8\u3002"),
                      ("long_prompt", read_json(TESTS / "extended.json")["prompt"])]
    require([(t["key"], t["prompt"]) for t in manifest["texts"]] == expected_texts and
            all(t["file"] == "text-" + t["key"] + ".safetensors" for t in manifest["texts"]),
            "Oracle text matrix differs")
    specs = case_specs()
    require([{k: case[k] for k in spec} for case, spec in zip(manifest["cases"], specs)] == specs and len(manifest["cases"]) == 7,
            "Oracle regression matrix differs")
    per_case = {"initial.safetensors", "final.safetensors", "video-decoded.safetensors", "audio-decoded.safetensors"}
    per_case.update(f"step-{step:02d}.safetensors" for step in range(20))
    expected = {case["key"] + "/" + filename for case in specs for filename in per_case}
    expected.update("text-" + key + ".safetensors" for key in ("english", "chinese", "long_prompt"))
    expected.add("tokenizer.json")
    require(manifest.get("units") is not None and manifest["units"]["directory"] == "units", "Fixed primitive/host oracle is required before candidate evaluation")
    unit_path = args.oracle / "units/manifest.json"
    require(digest(unit_path) == manifest["units"]["manifest_sha256"], "Unit oracle manifest changed")
    unit_manifest = validate_units(args.oracle / "units")
    expected.update("units/" + path for path in unit_manifest["files"])
    expected.add("units/manifest.json")
    require(set(manifest["files"]) == expected, "Missing or unexpected oracle tensor files")
    for name, wanted in manifest["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid oracle artifact path")
        require(digest(args.oracle / name) == wanted, "Frozen oracle artifact changed: " + name)
    return manifest


def run_test(args, report, script, *arguments):
    source = Path(script) if Path(script).is_absolute() else TESTS / script
    command = [sys.executable, str(source), *map(str, arguments)]
    name = source.stem
    log = args.output / (name + ".log")
    start = time.monotonic()
    entry = {"script": str(source), "command": command, "log": str(log), "passed": False}
    report.setdefault("subprocesses", []).append(entry)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=args.candidate_root, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=args.stage_timeout)
    entry.update(seconds=time.monotonic() - start, returncode=result.returncode, passed=result.returncode == 0)
    require(result.returncode == 0, f"{script} failed; see {log}")


def compare(report, label, got, expected, limits):
    import torch
    got, expected = got.detach().cpu().float(), expected.detach().cpu().float()
    if got.shape != expected.shape:
        raise NumericMismatch(f"{label}: shape differs {tuple(got.shape)} / {tuple(expected.shape)}")
    finite = bool(torch.isfinite(got).all() and torch.isfinite(expected).all())
    if finite:
        difference = got - expected
        relative = float(difference.norm() / expected.norm().clamp_min(1e-12))
        maximum = float(difference.abs().max() / expected.abs().max().clamp_min(1e-12))
        finite = math.isfinite(relative) and math.isfinite(maximum)
        if not finite:
            relative = maximum = None
    else:
        relative = maximum = None
    passed = finite and relative <= limits["relative_l2"] and maximum <= limits["normalized_max"]
    row = {"label": label, "passed": bool(passed), "finite": finite,
           "relative_l2": relative, "normalized_max": maximum, "limits": limits}
    report.setdefault("comparisons", []).append(row)
    return row


def configure_decoder(lib):
    pointer, size, integer = C.c_void_p, C.c_size_t, C.c_int
    lib.h3_phase1_video.argtypes = [C.c_char_p, pointer, integer, integer, integer, pointer, size, pointer, pointer, size]
    lib.h3_phase1_video.restype = integer
    lib.h3_phase1_audio.argtypes = [C.c_char_p, pointer, integer, pointer, size, pointer, pointer, size]
    lib.h3_phase1_audio.restype = integer


def check_tokenizer(args, report):
    cached = read_json(args.oracle / "tokenizer.json")["cases"]
    require(len(cached) == 10 and len({r["prompt"] for r in cached}) == 10, "Incomplete frozen tokenizer cases")
    lib = C.CDLL(str(args.candidate_root / "build-cuda/libh3_tokenizer.so"))
    pointer, size = C.c_void_p, C.c_size_t
    lib.h3_tokenizer_load.argtypes = [C.c_char_p, pointer, size]
    lib.h3_tokenizer_load.restype = pointer
    lib.h3_tokenizer_encode.argtypes = [pointer, C.c_char_p, C.c_int, C.POINTER(pointer), C.POINTER(size), pointer, size]
    lib.h3_tokenizer_encode.restype = C.c_int
    lib.h3_tokenizer_ids_free.argtypes, lib.h3_tokenizer_ids_free.restype = [pointer], None
    lib.h3_tokenizer_free.argtypes, lib.h3_tokenizer_free.restype = [pointer], None
    error = C.create_string_buffer(1024)
    tokenizer = lib.h3_tokenizer_load(str(args.weights / "tokenizer/tokenizer.json").encode(), error, len(error))
    require(bool(tokenizer), "Native tokenizer load: " + error.value.decode())
    report["tokenizer"] = []
    try:
        for row in cached:
            ids, count = pointer(), size()
            require(bool(lib.h3_tokenizer_encode(tokenizer, row["prompt"].encode(), 0, C.byref(ids), C.byref(count), error, len(error))),
                    "Native tokenizer encode: " + error.value.decode())
            try:
                got = list(C.cast(ids, C.POINTER(C.c_uint32))[:count.value])
            finally:
                lib.h3_tokenizer_ids_free(ids)
            report["tokenizer"].append({"prompt": row["prompt"], "native": got, "reference": row["reference"],
                                        "passed": got == row["reference"]})
    finally:
        lib.h3_tokenizer_free(tokenizer)
    require(all(row["passed"] for row in report["tokenizer"]), "Frozen tokenizer IDs differ")


def screen_case(args, report, case, session):
    torch, load_file, save_file, _, _, _, patchify = dependencies()
    from real_dit import CModel
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    limits = frozen_contract()["model"]
    root, output = args.candidate_root, args.output / case["key"]
    output.mkdir()
    oracle_root = args.oracle / case["key"]
    library = root / "build-cuda/libh3_model.so"
    spec = case["configuration"]
    initial = load_file(str(oracle_root / "initial.safetensors"))
    lib = C.CDLL(str(library))
    error = C.create_string_buffer(1024)
    candidate_text = session["texts"][case["text_key"]]
    compare(report, "text_embedding", candidate_text, initial["text"], limits["text_embedding"])
    save_file({"text": candidate_text}, str(output / "candidate-text.safetensors"))
    require(report["comparisons"][-1]["passed"], "Text embedding exceeds fixed limits")
    text_exact = same_bits(candidate_text, initial["text"])
    video, audio = initial["video"].clone(), initial["audio"].clone()
    geometry = (*video.shape[2:], audio.shape[-1], spec["frames"])
    model = None
    replay_steps, reused_steps = [], []
    native_forward_count = 0
    artifact_dir = output / "trajectory"
    artifact_dir.mkdir()

    def create(text):
        cached = session.get("native")
        if cached is not None and (session["geometry"] != geometry or not same_bits(session["native_text"], text)):
            cached.close()
            session.pop("native")
            cached = None
        native = cached if cached is not None else CModel(library, args.weights / "transformer", text, geometry,
                                                        initial["video_sigmas"], initial["audio_sigmas"])
        session.update(native=native, geometry=geometry, native_text=text.clone())
        native.reset()
        if not torch.equal(native.positions(), initial["positions"]):
            raise NumericMismatch("Exact positions differ")
        try:
            native.verify_layout_and_packing(initial["tags"], initial["video_indices"], initial["audio_indices"],
                                             initial["text_indices"], initial["video"], initial["audio"])
        except AssertionError as error:
            raise NumericMismatch("Exact layout or packing differs") from error
        return native

    def velocity_checks(step, velocity, audio_velocity, oracle):
        compare(report, f"step_{step}_video_velocity", patchify(velocity, (1, 2, 2)).unsqueeze(0),
                oracle["video_velocity_packed"], limits["dit_velocity"])
        compare(report, f"step_{step}_audio_velocity", audio_velocity.permute(0, 2, 3, 1).reshape(1, -1, 32),
                oracle["audio_velocity_packed"], limits["dit_velocity"])

    try:
        model = create(candidate_text)
        lib.h3_euler_velocity_step.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_float, C.c_float]
        lib.h3_euler_velocity_step.restype = C.c_int
        for step in range(20):
            oracle = load_file(str(oracle_root / f"step-{step:02d}.safetensors"))
            same_input = text_exact and same_bits(video, oracle["video_before"]) and same_bits(audio, oracle["audio_before"])
            velocity, audio_velocity = model.forward(step, video, audio)
            native_forward_count += 1
            if same_input:
                reused_steps.append(step)
                velocity_checks(step, velocity, audio_velocity, oracle)
            else:
                replay_steps.append(step)
            for modality, sample, value in (("video", video, velocity), ("audio", audio, audio_velocity)):
                sigmas = initial[modality + "_sigmas"]
                require(bool(lib.h3_euler_velocity_step(sample.data_ptr(), value.data_ptr(), sample.numel(),
                             sigmas[step].item(), sigmas[step + 1].item())), "Native Euler update failed")
                compare(report, f"step_{step}_{modality}_latent", sample, oracle[modality + "_after"],
                        limits["each_denoising_step"])
            save_file({"video": video, "audio": audio, "video_velocity": velocity, "audio_velocity": audio_velocity},
                      str(artifact_dir / f"step-{step:02d}.safetensors"))
            require(bool(torch.isfinite(video).all() and torch.isfinite(audio).all()), "Non-finite independent trajectory")
        if replay_steps:
            if text_exact:
                model.reset()
            else:
                model = create(initial["text"])
            for step in replay_steps:
                oracle = load_file(str(oracle_root / f"step-{step:02d}.safetensors"))
                velocity, audio_velocity = model.forward(step, oracle["video_before"], oracle["audio_before"])
                native_forward_count += 1
                velocity_checks(step, velocity, audio_velocity, oracle)
    finally:
        report["native_forward_count"] = native_forward_count
    save_file({"video": video, "audio": audio}, str(output / "candidate-latents.safetensors"))
    report.update(exact_checks={"positions": True, "token_tags": True, "packing_indices": True,
                  "latent_packing": True, "initial_noise": True}, text_exact=text_exact,
                  native_forward_count=native_forward_count, shared_same_input_velocity_steps=reused_steps,
                  additional_oracle_input_velocity_steps=replay_steps, independent_steps=20, layers=50)
    require(len(report["comparisons"]) == 81 and all(r["passed"] for r in report["comparisons"]),
            "Trajectory exceeds fixed limits; decoder work is not required for a failed candidate")
    configure_decoder(lib)
    final = load_file(str(oracle_root / "final.safetensors"))
    for modality, candidate_latent in (("video", video), ("audio", audio)):
        expected = load_file(str(oracle_root / f"{modality}-decoded.safetensors"))
        shape = (C.c_int * 3)()
        got = None
        for scope, latent in (("identical_latents", final[modality]), ("independent_trajectory", candidate_latent)):
            if got is None or not same_bits(candidate_latent, final[modality]):
                got = torch.empty_like(expected[scope])
                if modality == "video":
                    wrapper = read_json(args.weights / "video_vae/config.json")
                    ok = lib.h3_phase1_video(str(args.weights / "video_vae" / wrapper["source_path"]).encode(),
                        latent.data_ptr(), *latent.shape[2:], got.data_ptr(), got.numel(), shape, error, len(error))
                    expected_shape = tuple(got.shape[:3])
                else:
                    ok = lib.h3_phase1_audio(str(args.weights / "audio_vae").encode(), latent.data_ptr(),
                        latent.shape[-1], got.data_ptr(), got.numel(), shape, error, len(error))
                    expected_shape = (*got.shape, 32000)
                require(bool(ok), "Native decode: " + error.value.decode())
                if tuple(shape) != expected_shape:
                    raise NumericMismatch("Decoded metadata differs: " + modality)
            compare(report, modality + "_" + scope, got, expected[scope], limits[f"decoded_{modality}_float"])
            save_file({"native": got}, str(output / f"{modality}-{scope}.safetensors"))
    require(len(report["comparisons"]) == 85, "Missing fixed screening comparisons")
    require(all(r["passed"] for r in report["comparisons"]), "Fixed screening numeric limits exceeded")
    report["passed"] = True
    write_json(output / "results.json", report)


def evaluate(args, report, manifest):
    torch, load_file, save_file, *_ = dependencies()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    root, output = args.candidate_root, args.output
    library = root / "build-cuda/libh3_model.so"
    run_test(args, report, Path(__file__).with_name("frozen_primitives.py"), "replay",
             "--fixture-dir", args.oracle / "units", "--manifest-sha256", manifest["units"]["manifest_sha256"],
             "--cuda-library", root / "build-cuda/libh3_cuda.so", "--model-library", library,
             "--output", output / "units.json")
    units = read_json(output / "units.json")
    require(units.get("passed") is True and units.get("reference_recomputed") is False and
            units["primitives"]["passed"] is True and len(units["primitives"]["cases"]) >= 69 and
            units["host"]["passed"] is True and len(units["host"]["cases"]) == 12 and
            len(units["host"]["layout_cases"]) == 32, "Incomplete frozen primitive/host screen")
    check_tokenizer(args, report)
    lib = C.CDLL(str(library))
    lib.h3_phase1_text.argtypes = [C.c_char_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p, C.c_size_t]
    lib.h3_phase1_text.restype = C.c_int
    limits = frozen_contract()["model"]
    session = {"texts": {}}
    error = C.create_string_buffer(1024)
    report["cases"] = []
    try:
        for text_case in manifest["texts"]:
            if args.mode == "screen" and text_case["key"] != "long_prompt":
                continue
            initial = load_file(str(args.oracle / text_case["file"]))
            ids, expected = initial["token_ids"], initial["text"]
            got = torch.empty_like(expected)
            require(bool(lib.h3_phase1_text(str(args.weights / "text_encoder").encode(), ids.data_ptr(), ids.numel(),
                    got.data_ptr(), error, len(error))), "Native text: " + error.value.decode())
            compare(report, "text_" + text_case["key"], got, expected, limits["text_embedding"])
            save_file({"text": got}, str(output / (text_case["key"] + "-text.safetensors")))
            session["texts"][text_case["key"]] = got
        require(all(r["passed"] for r in report["comparisons"]), "Text embedding exceeded fixed limits")
        for case in manifest["cases"]:
            if args.mode == "screen" and case["key"] != "extended":
                continue
            case_report = {"key": case["key"], "configuration": case["configuration"], "passed": False}
            report["cases"].append(case_report)
            started = time.monotonic()
            try:
                screen_case(args, case_report, case, session)
            finally:
                case_report["wall_seconds_including_checks_and_io"] = time.monotonic() - started
                write_json(output / case["key"] / "results.json", case_report)
    finally:
        if "native" in session:
            session.pop("native").close()
    expected_cases = 1 if args.mode == "screen" else 7
    require(len(report["cases"]) == expected_cases and all(r["passed"] for r in report["cases"]), "Missing or failed cached trajectory")
    if args.mode == "promote":
        run_test(args, report, "cli_smoke.py", "--weights", args.weights, "--output", output)
        smoke = read_json(output / "cli-smoke.json")
        require(smoke.get("passed") is True and smoke["frames"] == 22 and smoke["steps"] == 20 and
                smoke["layers"] == 50 and smoke["decoded_audio_bytes"] > 0 and
                digest(output / "cli-smoke.mp4") == smoke["movie_sha256"], "Standalone CLI media check failed")
    report.update(promotion_eligible=args.mode == "promote", required_trajectories=6 if args.mode == "promote" else 0,
                  additional_trajectories=1,
                  reference_mode="Frozen PyTorch fixtures; no large PyTorch reference model is loaded or executed",
                  velocity_contract="Same-input candidate velocity at the fixed oracle states; no live candidate-state PyTorch reference. Independently evolving candidate latents are checked at every step; numeric limits unchanged.",
                  scope_note="Screening covers extended only; promotion covers all 6+1 cases, three text embeddings and standalone CLI media. Production orchestration/device-sampler changes outside the phase-one bridge require additional review.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "screen", "promote"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path, help="Accepted job 14274 full artifacts; prepare only")
    parser.add_argument("--unit-oracle", type=Path, help="One-time frozen_primitives.py capture directory; prepare only")
    parser.add_argument("--oracle", type=Path, help="Prepared, evaluator-owned cache")
    parser.add_argument("--oracle-sha256", help="Frozen manifest SHA256; mandatory for screen/promote")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--weights", type=Path, help="Original FL2VA directory")
    parser.add_argument("--diffusers", type=Path, help="Pinned source checkout")
    parser.add_argument("--stage-timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    started = time.monotonic()
    report = {"passed": False, "status": "FAIL", "mode": args.mode, "contract_sha256": CONTRACT_SHA256,
              "judge_sha256": digest(Path(__file__)), "promotion_eligible": False}
    writable = False
    try:
        require(not args.output.exists() or not any(args.output.iterdir()), "Output must be new or empty; stale evidence is rejected")
        args.output.mkdir(parents=True, exist_ok=True)
        writable = True
        require(args.stage_timeout > 0, "Stage timeout must be positive")
        for key in ("reference_root", "oracle", "candidate_root", "weights", "diffusers", "unit_oracle"):
            if getattr(args, key) is not None:
                setattr(args, key, getattr(args, key).resolve())
        if args.mode == "prepare":
            require(args.reference_root is not None and args.diffusers is not None, "prepare requires --reference-root and --diffusers")
            prepare(args, report)
        else:
            require(args.oracle is not None and args.oracle_sha256 is not None and args.candidate_root is not None and
                    args.weights is not None and args.diffusers is not None,
                    "screen/promote require oracle, its SHA256, candidate root, FL2VA weights and pinned --diffusers")
            check_reference_checkout(args.diffusers)
            manifest = load_manifest(args)
            names = ("h3", "libh3_model.so", "libh3_cuda.so", "libh3_tokenizer.so")
            report.update(oracle_manifest_sha256=args.oracle_sha256, configuration=manifest["configuration"],
                          candidate_binary_sha256={name: digest(args.candidate_root / "build-cuda" / name) for name in names})
            for key in ("H3_CUDA_TRACE_DIR", "H3_CUDA_TEXT_TRACE_DIR"):
                require(not os.environ.get(key), "Diagnostic environment is not allowed: " + key)
            evaluate(args, report, manifest)
        report.update(passed=True, status="PASS")
    except Exception as error:
        comparisons = report.get("comparisons", []) + report.get("tokenizer", [])
        cases = report.get("cases", [])
        comparisons += [row for case in (cases if isinstance(cases, list) else []) if isinstance(case, dict)
                        for row in case.get("comparisons", [])]
        mismatch = isinstance(error, NumericMismatch) or any(row.get("passed") is False for row in comparisons)
        report.update(error=str(error), traceback=traceback.format_exc(),
                      failure_kind="numeric_mismatch" if mismatch else "execution_or_infrastructure")
    finally:
        report["wall_seconds_including_checks_and_io"] = time.monotonic() - started
        if writable:
            write_json(args.output / "verification.json", report)
        else:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), file=sys.stderr)
    print(report["status"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
