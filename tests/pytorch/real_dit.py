"""Run original-checkpoint DiT parity against the pinned Diffusers forward.

Synthetic text is only a DiT diagnostic; it cannot satisfy end-to-end acceptance.
The saved input file is read by both implementations. No RNG is shared implicitly.
"""
import argparse
from contextlib import nullcontext
import ctypes as C
import importlib.util
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    video_latent_num_frames, audio_latent_num_frames,
)


def conversion_module(checkout):
    source = Path(checkout) / "scripts/convert_minimax_h3_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("official_h3_conversion", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(weights, conversion):
    config = conversion.MINIMAX_H3_TRANSFORMER_CONFIG
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**config)
    state = {}
    for path in sorted(Path(weights).glob("*.safetensors")):
        print("PyTorch loading", path.name, flush=True)
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                value = source.get_tensor(key)
                if key.endswith(".attn.qkv_proj.weight"):
                    value = conversion.reorder_interleaved_qkv(value, config["num_attention_heads"], config["attention_head_dim"])
                for target, tensor in conversion.convert_transformer_key(key, value, config):
                    if target in state:
                        raise ValueError("duplicate checkpoint tensor: " + target)
                    state[target] = tensor.to("cuda")
    model.load_state_dict(state, strict=True, assign=True)
    model.rope = MiniMaxH3RotaryPosEmbed(config["rope_freq_dim"], config["rope_theta"]).cuda()
    return model.eval()


class CModel:
    def __init__(self, library, weights, text, geometry, video_sigmas, audio_sigmas):
        self.lib = C.CDLL(str(Path(library).resolve()))
        pointer, integer, size = C.c_void_p, C.c_int, C.c_size_t
        self.lib.h3_phase1_create.argtypes = [C.c_char_p, pointer, size] + [integer] * 6 + [pointer, pointer, pointer, size]
        self.lib.h3_phase1_create.restype = pointer
        self.lib.h3_phase1_forward.argtypes = [pointer, integer] + [pointer] * 5 + [size]
        self.lib.h3_phase1_forward.restype = integer
        self.lib.h3_phase1_positions.argtypes = [pointer, pointer, size]
        self.lib.h3_phase1_positions.restype = size
        self.lib.h3_phase1_tags.argtypes = [pointer, pointer, size]
        self.lib.h3_phase1_tags.restype = size
        self.lib.h3_phase1_free.argtypes = [pointer]
        self.lib.h3_phase1_free.restype = None
        self.lib.h3_phase1_reset.argtypes = [pointer, pointer, size]
        self.lib.h3_phase1_reset.restype = integer
        self.error = C.create_string_buffer(1024)
        self.handle = self.lib.h3_phase1_create(
            str(weights).encode(), text.data_ptr(), text.shape[1], *geometry,
            len(video_sigmas) - 1, video_sigmas.data_ptr(), audio_sigmas.data_ptr(), self.error, len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode())

    def positions(self):
        count = self.lib.h3_phase1_positions(self.handle, None, 0)
        positions = torch.empty(count, dtype=torch.float64)
        self.lib.h3_phase1_positions(self.handle, positions.data_ptr(), count)
        return positions.reshape(-1, 3)

    def forward(self, step, video, audio):
        vo, ao = torch.empty_like(video), torch.empty_like(audio)
        if not self.lib.h3_phase1_forward(self.handle, step, video.data_ptr(), audio.data_ptr(), vo.data_ptr(), ao.data_ptr(), self.error, len(self.error)):
            raise RuntimeError(self.error.value.decode())
        return vo, ao

    def verify_layout_and_packing(self, tags, video_indices, audio_indices, text_indices, video, audio):
        got_tags = torch.empty_like(tags)
        self.lib.h3_phase1_tags(self.handle, got_tags.data_ptr(), got_tags.numel())
        torch.testing.assert_close(got_tags, tags, rtol=0, atol=0)
        for tag, indices in ((0, video_indices), (2, audio_indices), (1, text_indices)):
            torch.testing.assert_close(torch.where(got_tags == tag)[0], indices, rtol=0, atol=0)
        expected_video = patchify_video_latents(video, (1, 2, 2)).contiguous()
        expected_audio = audio.permute(0, 2, 3, 1).reshape(-1, 32).contiguous()
        for name, source, expected, dimensions in (
            ("h3_dit_patchify_video", video, expected_video, tuple(video.shape[1:])),
            ("h3_dit_pack_audio", audio, expected_audio, (audio.shape[1], audio.shape[-1])),
        ):
            function = getattr(self.lib, name)
            function.argtypes = [C.c_void_p] + [C.c_int] * len(dimensions) + [C.c_void_p, C.c_size_t]
            function.restype = C.c_int
            got = torch.empty_like(expected)
            if not function(source.data_ptr(), *dimensions, got.data_ptr(), got.numel()):
                raise RuntimeError(name + " failed")
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def close(self):
        self.lib.h3_phase1_free(self.handle)

    def reset(self):
        if not self.lib.h3_phase1_reset(self.handle, self.error, len(self.error)):
            raise RuntimeError(self.error.value.decode())


def main(argv=None, cache=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Original FL2VA directory")
    parser.add_argument("--diffusers", required=True, help="Pinned source checkout")
    parser.add_argument("--library", default="build-cuda/libh3_model.so")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", help="Saved BF16 text.safetensors with text embedding")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=20, help="Actual model evaluations")
    parser.add_argument("--evaluations", type=int, help="Diagnostic prefix only; never counts as full acceptance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true", help="Diagnostic first-step hidden-state snapshots")
    parser.add_argument("--attention-math-diagnostic", action="store_true", help="Diagnostic only: force PyTorch math SDPA")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.trace:
        (output / "trace-native").mkdir(exist_ok=True)
        os.environ["H3_CUDA_TRACE_DIR"] = str((output / "trace-native").resolve())
    else:
        os.environ.pop("H3_CUDA_TRACE_DIR", None)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    time_dim = video_latent_num_frames(args.frames, 17, 5)
    audio_dim = audio_latent_num_frames(args.frames)
    height, width = args.height // 16, args.width // 16
    text_file = load_file(args.text) if args.text else None
    text = text_file["text"] if text_file else torch.randn(1, 6, 5120).bfloat16()
    reference_text = text_file["reference_text"] if text_file else text
    inputs = {
        "text": text,
        "reference_text": reference_text.clone(),
        "video": torch.randn(1, 24, time_dim, height, width),
        "audio": torch.randn(1, 32, 2, audio_dim),
    }
    save_file(inputs, str(output / "inputs.safetensors"))
    inputs = load_file(str(output / "inputs.safetensors"))
    video_scheduler, audio_scheduler = MiniMaxH3Scheduler(shift=12), MiniMaxH3Scheduler(shift=3)
    for scheduler in (video_scheduler, audio_scheduler):
        scheduler.set_timesteps(args.steps + 1)
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(text.shape[1], dtype=torch.long), time_dim, height, width,
        audio_dim, (1, 2, 2), 2, 2, 0)
    positions, tags, video_indices, audio_indices, text_indices = layout[:5]
    # Use the public pipeline's row plan, including its single distinct timestep
    # at the first evaluation. Duplicate timestep rows can select different GEMMs.
    row_timestep_plan = [
        tuple(tensor.cuda() for tensor in MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices, audio_indices, 0, 0, text_indices.numel(),
            float(vt), float(at), max(float(vt), 0.999), 1.0))
        for vt, at in zip(video_scheduler.timesteps, audio_scheduler.timesteps)
    ]
    kwargs = dict(position_ids=positions.cuda(), token_tags=tags.cuda(), video_indices=video_indices.cuda(),
                  audio_indices=audio_indices.cuda(), text_indices=text_indices.cuda(),
                  encoder_hidden_states=inputs["text"].cuda())
    started = time.monotonic()
    cache = cache if cache is not None else {}
    key = (args.weights, args.text, time_dim, height, width, audio_dim, args.steps)
    if cache.get("native_key") != key:
        if "native" in cache:
            cache.pop("native").close()
        cache["native"] = CModel(args.library, Path(args.weights) / "transformer", inputs["text"],
                        (time_dim, height, width, audio_dim, args.frames), video_scheduler.sigmas, audio_scheduler.sigmas)
        cache["native_key"] = key
    native = cache["native"]
    native.reset()
    native_load_seconds = time.monotonic() - started
    torch.testing.assert_close(native.positions(), positions, rtol=0, atol=0)
    native.verify_layout_and_packing(tags, video_indices, audio_indices, text_indices, inputs["video"], inputs["audio"])
    started = time.monotonic()
    if "reference" not in cache:
        cache["reference"] = load_reference(Path(args.weights) / "transformer", conversion_module(args.diffusers))
    reference = cache["reference"]
    reference_load_seconds = time.monotonic() - started
    thresholds = json.loads(Path(__file__).with_name("acceptance.json").read_text())["model"]
    records = []
    video, audio = inputs["video"].clone(), inputs["audio"].clone()
    ref_video, ref_audio = video.cuda(), audio.cuda()
    native.lib.h3_euler_velocity_step.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_float, C.c_float]
    native.lib.h3_euler_velocity_step.restype = C.c_int

    def unpack_video(rows):
        return rows.reshape(1, time_dim, height // 2, width // 2, 24, 1, 2, 2).permute(
            0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, time_dim, height, width).contiguous()

    def unpack_audio(rows):
        return rows.reshape(1, 2, audio_dim, 32).permute(0, 3, 1, 2).contiguous()

    def compare(label, got, expected, kind):
        got, expected = got.float().cpu(), expected.float().cpu()
        if got.shape != expected.shape:
            raise ValueError(f"{label}: native shape {got.shape}, reference shape {expected.shape}")
        delta = got - expected
        rel = float(delta.norm() / expected.norm().clamp_min(1e-12))
        normmax = float(delta.abs().max() / expected.abs().max().clamp_min(1e-12))
        passed = bool(torch.isfinite(got).all() and torch.isfinite(expected).all() and
                      rel <= thresholds[kind]["relative_l2"] and normmax <= thresholds[kind]["normalized_max"])
        row = dict(label=label, relative_l2=rel, normalized_max=normmax, passed=passed)
        records.append(row)
        print(json.dumps(row), flush=True)

    timings = []
    from torch.nn.attention import sdpa_kernel, SDPBackend
    attention_context = sdpa_kernel(SDPBackend.MATH) if args.attention_math_diagnostic else nullcontext()
    traces, hooks = {}, []
    if args.trace:
        def capture(label):
            def hook(module, arguments, output):
                traces[label] = output.detach().cpu().contiguous()
            return hook
        def capture_input(module, arguments, kwargs):
            value = arguments[0] if arguments else kwargs["hidden_states"]
            traces["packed_input"] = value.detach().cpu().contiguous()
        hooks.append(reference.transformer_blocks[0].register_forward_pre_hook(capture_input, with_kwargs=True))
        for index, block in enumerate(reference.transformer_blocks):
            hooks.append(block.register_forward_hook(capture(f"block-{index:02d}")))
    with torch.inference_mode(), attention_context:
        for step in range(args.evaluations or args.steps):
            vt, at = video_scheduler.timesteps[step], audio_scheduler.timesteps[step]
            kwargs["timestep"], kwargs["timestep_indices"] = row_timestep_plan[step]
            # Velocity parity uses exactly the same saved native input at this step.
            torch.cuda.synchronize()
            started = time.monotonic()
            expected = reference(hidden_states=patchify_video_latents(video.cuda(), (1, 2, 2)).unsqueeze(0),
                                 audio_hidden_states=audio.cuda().permute(0, 2, 3, 1).reshape(1, -1, 32), **kwargs)
            torch.cuda.synchronize()
            reference_seconds = time.monotonic() - started
            if hooks:
                for hook in hooks:
                    hook.remove()
                hooks.clear()
                save_file(traces, str(output / "trace-reference.safetensors"))
            trajectory_kwargs = {**kwargs, "encoder_hidden_states": inputs["reference_text"].cuda()}
            trajectory = expected if step == 0 and torch.equal(text, reference_text) else reference(
                hidden_states=patchify_video_latents(ref_video, (1, 2, 2)).unsqueeze(0),
                audio_hidden_states=ref_audio.permute(0, 2, 3, 1).reshape(1, -1, 32), **trajectory_kwargs)
            torch.cuda.synchronize()
            started = time.monotonic()
            velocity, audio_velocity = native.forward(step, video, audio)
            native_seconds = time.monotonic() - started
            packed_video = patchify_video_latents(velocity, (1, 2, 2)).unsqueeze(0)
            packed_audio = audio_velocity.permute(0, 2, 3, 1).reshape(1, -1, 32)
            compare(f"step_{step}_video_velocity", packed_video, expected.sample, "dit_velocity")
            compare(f"step_{step}_audio_velocity", packed_audio, expected.audio_sample, "dit_velocity")
            timings.append(dict(step=step, native_seconds=native_seconds, reference_seconds=reference_seconds))
            save_file({"video_velocity": velocity, "audio_velocity": audio_velocity,
                       "reference_video_packed": expected.sample.cpu().contiguous(),
                       "reference_audio_packed": expected.audio_sample.cpu().contiguous()},
                      str(output / f"step-{step:02d}.safetensors"))
            for sample, v, scheduler in ((video, velocity, video_scheduler), (audio, audio_velocity, audio_scheduler)):
                ok = native.lib.h3_euler_velocity_step(sample.data_ptr(), v.data_ptr(), sample.numel(),
                    scheduler.sigmas[step].item(), scheduler.sigmas[step + 1].item())
                if not ok:
                    raise RuntimeError("Native Euler update failed")
            ref_video = video_scheduler.step(unpack_video(trajectory.sample), vt, ref_video).prev_sample
            ref_audio = audio_scheduler.step(unpack_audio(trajectory.audio_sample), at, ref_audio).prev_sample
            compare(f"step_{step}_video_latent", video, ref_video, "each_denoising_step")
            compare(f"step_{step}_audio_latent", audio, ref_audio, "each_denoising_step")
    save_file({"video": video, "audio": audio, "reference_video": ref_video.cpu().contiguous(),
               "reference_audio": ref_audio.cpu().contiguous()}, str(output / "latents.safetensors"))
    report = dict(scope="DiT and sampler; text and decoded-output acceptance remain separate", arguments=vars(args),
                  complete_trajectory=(args.evaluations is None or args.evaluations == args.steps),
                  real_text=bool(args.text), layers=50,
                  diagnostic_attention=args.attention_math_diagnostic, trace_enabled=args.trace,
                  reference_timestep_plan="MiniMaxH3SetTimestepsStep.build_row_timesteps",
                  reference_timestep_rows=[times.numel() for times, _ in row_timestep_plan],
                  exact_checks=dict(positions=True, token_tags=True, packing_indices=True, latent_packing=True),
                  input_sha256=hashlib.sha256((output / "inputs.safetensors").read_bytes()).hexdigest(),
                  passed=all(r["passed"] for r in records), results=records, timings=timings,
                  load_seconds=dict(native=native_load_seconds, reference=reference_load_seconds))
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    session = {}
    try:
        report = main(cache=session)
    finally:
        if "native" in session:
            session.pop("native").close()
    raise SystemExit(0 if report["passed"] else 1)
