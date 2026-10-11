#!/usr/bin/env python3
"""Capture primitive/host PyTorch references once; replay only native C calls.

capture requires the pinned PyTorch/Diffusers environment and CUDA. replay uses
only ctypes and NumPy; it never imports or executes a PyTorch reference operator.
The replay caller must supply the previously recorded manifest SHA256.
"""
from __future__ import annotations

import argparse
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback
import types


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ctype(name):
    if "*" in name:
        return C.c_void_p
    return {"int": C.c_int, "float": C.c_float, "size_t": C.c_size_t,
            "uint32_t": C.c_uint32, "uint64_t": C.c_uint64,
            "h3_gpu_dtype": C.c_int, "void": None}[name]


def capture_abi(root):
    source = (root / "h3_gpu.h").read_text() + (root / "cuda/h3_cuda.h").read_text()
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return {
        name: {"result": result.strip(), "args": [
            a.strip().rsplit(" ", 1)[0] + (" *" if "*" in a.rsplit(" ", 1)[-1] else "")
            for a in args.split(",") if a.strip() != "void"]}
        for result, name, args in re.findall(
            r"([\w ]+\s*\*?)\s*(h3_gpu_\w+)\s*\(([^;]+)\);", source)
    }


class CaptureCheck(ast.NodeTransformer):
    """Replace only the existing test's local check body, retaining its cases."""
    def __init__(self):
        self.replaced = 0

    def visit_FunctionDef(self, node):
        if node.name == "check":
            names = [arg.arg for arg in node.args.args]
            if names != ["name", "shape", "dtype", "function", "inputs", "scalars", "reference"]:
                raise ValueError("Unexpected primitives.check signature")
            node.body = [ast.Return(ast.Call(
                func=ast.Name(id="_capture_check", ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load()) for name in ["records", *names]],
                keywords=[]))]
            self.replaced += 1
            return node
        return self.generic_visit(node)


def capture(output_dir, tests_dir):
    import torch
    import diffusers
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        video_latent_num_frames, audio_latent_num_frames)

    output = Path(output_dir).resolve()
    tests = Path(tests_dir).resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing frozen oracle")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".capture-", dir=output.parent))
    files = {}
    primitive_cases = []

    def tensor(value):
        value = value.detach().cpu().contiguous()
        name = f"tensor-{len(files):05d}.bin"
        raw = value.view(torch.uint8).numpy().tobytes()
        (temporary / name).write_bytes(raw)
        files[name] = sha(raw)
        return {"file": name, "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch.")}

    def check(records, name, shape, dtype, function, inputs, scalars, reference):
        with torch.no_grad():
            expected = reference(*[x.cuda() for x in inputs]).detach().cpu().float()
        primitive_cases.append({"name": name, "shape": list(shape),
                               "dtype": str(dtype).removeprefix("torch."),
                               "function": function, "inputs": [tensor(x) for x in inputs],
                               "scalars": scalars, "expected": tensor(expected)})
        records.append({"name": name, "passed": True})

    class NoNative:
        def __init__(self, *_args):
            pass

        def close(self):
            pass

    try:
        # Original main supplies all 69 shapes, seed values, dtypes, and reference
        # functions. It computes no native results while creating this oracle.
        source = tests / "primitives.py"
        transform = CaptureCheck()
        tree = transform.visit(ast.parse(source.read_text(), filename=str(source)))
        if transform.replaced != 1:
            raise ValueError("Expected exactly one primitives.check function")
        ast.fix_missing_locations(tree)
        module = types.ModuleType("_frozen_primitive_capture")
        module.__file__ = str(source)
        sys.path.insert(0, str(tests))
        previous_argv = sys.argv
        try:
            exec(compile(tree, str(source), "exec"), module.__dict__)
            module.Native = NoNative
            module._capture_check = check
            sys.argv = [str(source), "--output", str(temporary / "capture-report.json")]
            try:
                module.main()
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
        finally:
            sys.argv = previous_argv
            sys.path.pop(0)

        # Preserve the host test's seed and order. No native library is loaded.
        torch.manual_seed(20261010)
        host_cases = []
        for steps in (2, 3, 7, 19, 20, 1000):
            for modality, shift in (("video", 12), ("audio", 3)):
                scheduler = MiniMaxH3Scheduler(shift=shift)
                scheduler.set_timesteps(steps + 1)
                initial = torch.randn(259)
                expected = initial.clone()
                velocities, updates = [], []
                for step in range(steps):
                    velocity = torch.randn_like(initial)
                    expected = scheduler.step(velocity, scheduler.timesteps[step], expected).prev_sample
                    velocities.append(velocity)
                    updates.append(expected)
                host_cases.append({"modality": modality, "steps": steps,
                                   "sigmas": tensor(scheduler.sigmas), "initial": tensor(initial),
                                   "velocities": tensor(torch.stack(velocities)),
                                   "expected_updates": tensor(torch.stack(updates))})
        layout_cases = []
        for tokens in (6, 8, 11, 82):
            for width, height in ((256, 256), (512, 512), (864, 480), (480, 864)):
                for frames in (22, 56):
                    latent_t = video_latent_num_frames(frames, 17, 5)
                    audio_t = audio_latent_num_frames(frames)
                    expected = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                        torch.ones(tokens, dtype=torch.long), latent_t, height // 16, width // 16,
                        audio_t, (1, 2, 2), 2, 2, 0)[0]
                    layout_cases.append({"tokens": tokens, "width": width, "height": height,
                                         "frames": frames, "latent_t": latent_t, "audio_t": audio_t,
                                         "expected_positions": tensor(expected)})
        if (len(primitive_cases), len(host_cases), len(layout_cases)) != (69, 12, 32):
            raise ValueError("Unexpected case count; explicitly review the capture contract")
        report = temporary / "capture-report.json"
        files[report.name] = sha(report.read_bytes())
        root = tests.parent.parent
        manifest = {
            "schema_version": 1, "primitive_cases": primitive_cases,
            "host_cases": host_cases, "layout_cases": layout_cases,
            "limits": json.loads((tests / "acceptance.json").read_text())["primitives"],
            "abi": capture_abi(root), "files": files,
            "provenance": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                           "device": torch.cuda.get_device_name(),
                           "capture_source_sha256": sha(Path(__file__).read_bytes()),
                           "source_sha256": {name: sha((tests / name).read_bytes())
                                             for name in ("primitives.py", "host.py", "native.py",
                                                          "acceptance.json", "sources.json")}}}
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
        return {"manifest_sha256": sha((output / "manifest.json").read_bytes()),
                "fixture_dir": str(output), "primitive_cases": 69,
                "sampler_cases": 12, "layout_cases": 32}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_manifest(directory, expected_sha):
    root = Path(directory).resolve()
    raw = (root / "manifest.json").read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError("Frozen manifest SHA256 mismatch")
    manifest = json.loads(raw)
    if manifest["schema_version"] != 1:
        raise ValueError("Unknown frozen oracle schema")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.resolve().parent != root or sha(path.read_bytes()) != expected:
            raise ValueError(f"Frozen tensor/file hash mismatch: {name}")
    return manifest


def read_tensor(root, item):
    import numpy as np
    dtypes = {"float32": "<f4", "float64": "<f8", "int32": "<i4",
              "int64": "<i8", "bfloat16": "<u2"}
    return np.frombuffer((root / item["file"]).read_bytes(), dtype=dtypes[item["dtype"]]).copy().reshape(item["shape"])


def as_float(array, dtype):
    import numpy as np
    return ((array.astype(np.uint32) << 16).view(np.float32)
            if dtype == "bfloat16" else array.astype(np.float32, copy=False))


class NativeReplay:
    def __init__(self, library, abi):
        self.lib = C.CDLL(str(Path(library).resolve()))
        for name, spec in abi.items():
            fn = getattr(self.lib, name)
            fn.restype = ctype(spec["result"])
            fn.argtypes = [ctype(arg) for arg in spec["args"]]
        error = C.create_string_buffer(512)
        self.gpu = self.lib.h3_gpu_create(None, error, len(error))
        self.owned = []
        if not self.gpu:
            raise RuntimeError(error.value.decode())

    def call(self, name, *args):
        result = getattr(self.lib, "h3_gpu_" + name)(self.gpu, *args)
        if not result:
            error = self.lib.h3_gpu_error(self.gpu)
            raise RuntimeError(name + ": " + C.string_at(error).decode())
        return result

    def tensor(self, value, dtype):
        suffix = {"float32": "f32", "bfloat16": "bf16", "int32": "u32"}[dtype]
        result = self.call("tensor_from_" + suffix, value.ctypes.data, value.size)
        self.owned.append(result)
        return result

    def clear(self):
        for pointer in self.owned:
            self.lib.h3_gpu_tensor_free(pointer)
        self.owned.clear()

    def close(self):
        self.clear()
        self.lib.h3_gpu_free(self.gpu)


def primitive_replay(root, manifest, library):
    import numpy as np
    native = NativeReplay(library, manifest["abi"])
    records = []
    try:
        for case in manifest["primitive_cases"]:
            row = {"name": case["name"], "shape": case["shape"], "dtype": case["dtype"]}
            try:
                actual = np.zeros(case["shape"], dtype=np.uint16 if case["dtype"] == "bfloat16" else np.float32)
                output = native.tensor(actual, case["dtype"])
                inputs = [native.tensor(read_tensor(root, item), item["dtype"]) for item in case["inputs"]]
                native.call(case["function"], output, *inputs, *case["scalars"])
                native.call("submit")
                suffix = "bf16" if case["dtype"] == "bfloat16" else "f32"
                if not getattr(native.lib, "h3_gpu_tensor_read_" + suffix)(output, actual.ctypes.data, actual.size):
                    raise RuntimeError("Native output read failed")
                actual = as_float(actual, case["dtype"])
                expected = read_tensor(root, case["expected"])
                difference = actual - expected
                relative = float(np.linalg.norm(difference.reshape(-1)) /
                                 max(float(np.linalg.norm(expected.reshape(-1))), 1e-12))
                limits = manifest["limits"][case["dtype"]]
                finite = bool(np.isfinite(actual).all())
                passed = finite and bool(np.allclose(actual, expected, atol=limits["atol"],
                                                     rtol=limits["rtol"])) and relative <= limits["relative_l2"]
                row.update(passed=passed, max_abs=float(np.abs(difference).max()) if finite else None,
                           relative_l2=relative if np.isfinite(relative) else None)
            except Exception:
                row.update(passed=False, error=traceback.format_exc())
            finally:
                native.clear()
            records.append(row)
    finally:
        native.close()
    return {"passed": all(row["passed"] for row in records), "cases": records}


class Schedule(C.Structure):
    _fields_ = [("steps", C.c_int), ("video", C.c_float * 1001), ("audio", C.c_float * 1001)]


class LayoutSpec(C.Structure):
    _fields_ = [(name, C.c_int) for name in
                ("text_len", "latent_t", "latent_h", "latent_w", "audio_t", "frame_count")] + [
        ("keyframes", C.c_void_p), ("keyframe_count", C.c_size_t),
        ("references", C.c_void_p), ("reference_count", C.c_size_t)]


class Layout(C.Structure):
    _fields_ = [("seq_len", C.c_size_t), ("segments", C.c_void_p), ("segment_count", C.c_size_t),
                ("positions", C.c_void_p)] + [(name, C.c_size_t) for name in (
                    "img_cond_rows", "img_target_rows", "audio_cond_rows", "audio_target_rows")] + [
        ("signature", C.c_int * 5)]


def host_replay(root, manifest, library):
    import numpy as np
    lib = C.CDLL(str(Path(library).resolve()))
    lib.h3_serving_schedule_build.argtypes = [C.c_int, C.POINTER(Schedule)]
    lib.h3_euler_velocity_step.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_float, C.c_float]
    lib.h3_layout_build.argtypes = [C.POINTER(LayoutSpec), C.POINTER(Layout), C.c_void_p, C.c_size_t]
    lib.h3_layout_free.argtypes = [C.POINTER(Layout)]
    records, layouts = [], []
    for case in manifest["host_cases"]:
        row = {"modality": case["modality"], "steps": case["steps"]}
        try:
            schedule = Schedule()
            if not lib.h3_serving_schedule_build(case["steps"], C.byref(schedule)):
                raise RuntimeError("Native schedule failed")
            sigmas = np.ctypeslib.as_array(getattr(schedule, case["modality"]))[:case["steps"] + 1]
            expected_sigmas = read_tensor(root, case["sigmas"])
            sigma_exact = sigmas.tobytes() == expected_sigmas.tobytes()
            sample = read_tensor(root, case["initial"])
            velocities = read_tensor(root, case["velocities"])
            updates = read_tensor(root, case["expected_updates"])
            failures = []
            for step in range(case["steps"]):
                velocity = velocities[step]
                if not lib.h3_euler_velocity_step(sample.ctypes.data, velocity.ctypes.data, sample.size,
                                                 float(sigmas[step]), float(sigmas[step + 1])):
                    raise RuntimeError("Native Euler update failed")
                if sample.tobytes() != updates[step].tobytes():
                    failures.append(step)
            row.update(sigma_exact=sigma_exact, updates_exact=not failures,
                       failed_update_steps=failures, passed=sigma_exact and not failures)
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
        records.append(row)
    for case in manifest["layout_cases"]:
        row = {key: case[key] for key in ("tokens", "width", "height", "frames")}
        layout = Layout()
        try:
            spec = LayoutSpec(case["tokens"], case["latent_t"], case["height"] // 16,
                              case["width"] // 16, case["audio_t"], case["frames"], None, 0, None, 0)
            error = C.create_string_buffer(1024)
            if not lib.h3_layout_build(C.byref(spec), C.byref(layout), error, len(error)):
                raise RuntimeError(error.value.decode())
            expected = read_tensor(root, case["expected_positions"])
            same_shape = expected.shape == (layout.seq_len, 3)
            row["passed"] = same_shape and C.string_at(layout.positions, expected.nbytes) == expected.tobytes()
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
        finally:
            if layout.positions:
                lib.h3_layout_free(C.byref(layout))
        layouts.append(row)
    return {"passed": all(row["passed"] for row in records + layouts),
            "cases": records, "layout_cases": layouts}


def replay(fixture_dir, manifest_sha256, cuda_library, model_library, output):
    root = Path(fixture_dir).resolve()
    result = {"passed": False, "reference_recomputed": False,
              "manifest_sha256": manifest_sha256}
    try:
        manifest = load_manifest(root, manifest_sha256)
        if tuple(len(manifest[key]) for key in ("primitive_cases", "host_cases", "layout_cases")) != (69, 12, 32):
            raise ValueError("Incomplete unit oracle")
        result["primitives"] = primitive_replay(root, manifest, cuda_library)
        result["host"] = host_replay(root, manifest, model_library)
        result["passed"] = result["primitives"]["passed"] and result["host"]["passed"]
    except Exception:
        result["error"] = traceback.format_exc()
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("capture")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--tests-dir", required=True)
    check = commands.add_parser("replay")
    check.add_argument("--fixture-dir", required=True)
    check.add_argument("--manifest-sha256", required=True)
    check.add_argument("--cuda-library", required=True)
    check.add_argument("--model-library", required=True)
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture(args.output_dir, args.tests_dir)
    else:
        result = replay(args.fixture_dir, args.manifest_sha256, args.cuda_library,
                        args.model_library, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
