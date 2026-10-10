"""Exact production sigma-grid and FP32 Euler comparisons against PyTorch."""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames, audio_latent_num_frames


class Schedule(C.Structure):
    _fields_ = [("steps", C.c_int), ("video", C.c_float * 1001), ("audio", C.c_float * 1001)]


class LayoutSpec(C.Structure):
    _fields_ = [(name, C.c_int) for name in ("text_len", "latent_t", "latent_h", "latent_w", "audio_t", "frame_count")] + [
        ("keyframes", C.c_void_p), ("keyframe_count", C.c_size_t),
        ("references", C.c_void_p), ("reference_count", C.c_size_t)]


class Layout(C.Structure):
    _fields_ = [("seq_len", C.c_size_t), ("segments", C.c_void_p), ("segment_count", C.c_size_t),
                ("positions", C.c_void_p)] + [(name, C.c_size_t) for name in (
                    "img_cond_rows", "img_target_rows", "audio_cond_rows", "audio_target_rows")] + [("signature", C.c_int * 5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build-cuda/libh3_model.so")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    library = C.CDLL(str(Path(args.library).resolve()))
    library.h3_serving_schedule_build.argtypes = [C.c_int, C.POINTER(Schedule)]
    library.h3_euler_velocity_step.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_float, C.c_float]
    torch.manual_seed(20261010)
    records = []
    for steps in (2, 3, 7, 19, 20, 1000):
        schedule = Schedule()
        if not library.h3_serving_schedule_build(steps, C.byref(schedule)):
            raise RuntimeError("Native schedule construction failed")
        for modality, shift in (("video", 12), ("audio", 3)):
            reference = MiniMaxH3Scheduler(shift=shift)
            reference.set_timesteps(steps + 1)
            native_sigmas = torch.tensor(list(getattr(schedule, modality))[:steps + 1])
            sigma_exact = torch.equal(native_sigmas, reference.sigmas)
            sample = torch.randn(259)
            expected = sample.clone()
            updates_exact = True
            for step in range(steps):
                velocity = torch.randn_like(sample)
                if not library.h3_euler_velocity_step(sample.data_ptr(), velocity.data_ptr(), sample.numel(),
                        native_sigmas[step].item(), native_sigmas[step + 1].item()):
                    raise RuntimeError("Native sampler failed")
                expected = reference.step(velocity, reference.timesteps[step], expected).prev_sample
                updates_exact = updates_exact and torch.equal(sample, expected)
            row = dict(modality=modality, steps=steps, sigma_exact=sigma_exact, updates_exact=updates_exact,
                       max_sigma_error=float((native_sigmas-reference.sigmas).abs().max()),
                       passed=sigma_exact and updates_exact)
            records.append(row)
            print(json.dumps(row), flush=True)
    layout_records = []
    library.h3_layout_build.argtypes = [C.POINTER(LayoutSpec), C.POINTER(Layout), C.c_void_p, C.c_size_t]
    library.h3_layout_free.argtypes = [C.POINTER(Layout)]
    for tokens in (6, 8, 11, 82):
        for width, height in ((256, 256), (512, 512), (864, 480), (480, 864)):
            for frames in (22, 56):
                latent_t = video_latent_num_frames(frames, 17, 5)
                audio_t = audio_latent_num_frames(frames)
                spec = LayoutSpec(tokens, latent_t, height//16, width//16, audio_t, frames, None, 0, None, 0)
                layout, error = Layout(), C.create_string_buffer(1024)
                if not library.h3_layout_build(C.byref(spec), C.byref(layout), error, len(error)):
                    raise RuntimeError(error.value.decode())
                try:
                    raw = bytearray(C.string_at(layout.positions, layout.seq_len*3*8))
                    native_positions = torch.frombuffer(raw, dtype=torch.float64).reshape(-1,3)
                    expected_positions = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                        torch.ones(tokens,dtype=torch.long), latent_t, height//16, width//16, audio_t,
                        (1,2,2), 2, 2, 0)[0]
                    row = dict(tokens=tokens, width=width, height=height, frames=frames,
                               passed=torch.equal(native_positions, expected_positions))
                    layout_records.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    library.h3_layout_free(C.byref(layout))
    result = dict(passed=all(r["passed"] for r in records+layout_records), cases=records, layout_cases=layout_records)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
