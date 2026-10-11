"""Video-VAE precision study: decode the same latents in several precisions
and measure each against the PyTorch FP32 decode.

PyTorch rows:
  torch-fp32      FP32 weights and compute (the truth for this study)
  torch-fp16      diffusers mixed precision: from_pretrained(dtype=float16)
                  keeps the `_keep_in_fp32_modules` (norms, scales) in FP32,
                  the projections and attention run in FP16
  torch-bf16      the same with bfloat16
  torch-autocast  FP32 weights under torch.autocast(float16) (vLLM-Omni)
Native rows (H3_VAE_PRECISION): h3-fp32, h3-3xtf32, h3-fp16.

Acceptance for h3-fp16: finite output, and on every case its relative L2
and its normalized maximum error against torch-fp32 are no larger than those
of torch-fp16, the diffusers FP16 decode.
"""
import argparse
import ctypes as C
import copy
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
from real_dit import conversion_module

NATIVE = {"h3-fp32": "fp32", "h3-3xtf32": "3xtf32", "h3-fp16": "fp16"}


def load_cases(args):
    cases = []
    for item in args.trajectory or []:
        path = Path(item)
        cases.append((path.parent.name if path.is_file() else path.name,
                      load_file(str(path if path.is_file() else path / "latents.safetensors"))["video"]))
    for item in args.raw or []:
        # name=path/video_latent.f32:T:H:W (normalized latent, [24][T][H][W])
        name, spec = item.split("=", 1)
        path, t, h, w = spec.rsplit(":", 3)
        data = np.fromfile(path, dtype=np.float32)
        cases.append((name, torch.from_numpy(data).reshape(1, 24, int(t), int(h), int(w))))
    return cases


def build_model(root, conversion):
    config = dict(conversion.MINIMAX_H3_VIDEO_VAE_CONFIG)
    wrapper = json.loads((root / "video_vae/config.json").read_text())
    config.update({k: wrapper[k] for k in ("latents_mean", "latents_std")})
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3(**config)
    state = {}
    source_path = root / "video_vae" / wrapper["source_path"] / wrapper["source_safetensors_path"]
    with safe_open(source_path, framework="pt", device="cpu") as source:
        for key in source.keys():
            for target, value in conversion.convert_video_vae_key(key, source.get_tensor(key), config):
                state[target] = value.cuda()
    model.load_state_dict(state, strict=True, assign=True)
    model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(
        int(config["decoder_attention_head_dim"] * config["decoder_rope_dim_ratio"]),
        theta=config["decoder_rope_theta"]).cuda()
    model.eval()
    return model, config, str(root / "video_vae" / wrapper["source_path"])


def mixed_precision(model, dtype):
    """What from_pretrained(dtype=...) produces: every parameter in `dtype`
    except modules named in `_keep_in_fp32_modules`."""
    keep = set(model._keep_in_fp32_modules or [])
    copy_ = copy.deepcopy(model)
    for name, parameter in copy_.named_parameters():
        if parameter.dtype.is_floating_point and not keep.intersection(name.split(".")):
            parameter.data = parameter.data.to(dtype)
    for name, buffer in copy_.named_buffers():
        if buffer.dtype.is_floating_point and not keep.intersection(name.split(".")) \
                and not name.startswith("decoder.rope"):
            buffer.data = buffer.data.to(dtype)
    return copy_


def torch_decode(model, config, latent, autocast=False):
    mean = torch.tensor(config["latents_mean"], device="cuda").view(1, 24, 1, 1, 1)
    std = torch.tensor(config["latents_std"], device="cuda").view(1, 24, 1, 1, 1)
    pixel_mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1, 1)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        decoded = model.decode(latent.cuda() * std + mean).sample
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    rgb = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1).squeeze(0).permute(1, 2, 3, 0)
    return rgb.cpu(), seconds


def metrics(got, truth):
    got64, truth64 = got.double(), truth.double()
    delta = got64 - truth64
    eight = (got64 * 255).round() - (truth64 * 255).round()
    mse = float(delta.square().mean())
    return dict(
        finite=bool(torch.isfinite(got).all()),
        relative_l2=float(delta.norm() / truth64.norm()),
        normalized_max=float(delta.abs().max() / truth64.abs().max()),
        psnr_db=float("inf") if mse == 0 else 10 * math.log10(1.0 / mse),
        pixels_equal=float((eight == 0).double().mean()),
        pixels_off_by_1=float((eight.abs() == 1).double().mean()),
        pixels_off_by_more=float((eight.abs() > 1).double().mean()),
        max_8bit=int(eight.abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--trajectory", action="append")
    parser.add_argument("--raw", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conversion = conversion_module(args.diffusers)
    library = C.CDLL(str(Path("build-cuda/libh3_model.so").resolve()))
    pointer, size, integer = C.c_void_p, C.c_size_t, C.c_int
    library.h3_phase1_video.argtypes = [C.c_char_p, pointer, integer, integer, integer, pointer,
                                        size, pointer, pointer, size]
    library.h3_phase1_video.restype = integer
    error = C.create_string_buffer(1024)
    shape = (integer * 3)()
    cases = load_cases(args)
    rows = []
    with torch.inference_mode():
        model, config, native_weights = build_model(Path(args.weights), conversion)
        variants = {"torch-fp16": mixed_precision(model, torch.float16),
                    "torch-bf16": mixed_precision(model, torch.bfloat16)}
        for name, latent in cases:
            truth, seconds = torch_decode(model, config, latent)
            results = {"torch-fp32": (truth, seconds)}
            for label, variant in variants.items():
                results[label] = torch_decode(variant, config, latent)
            results["torch-autocast"] = torch_decode(model, config, latent, autocast=True)
            for label, mode in NATIVE.items():
                os.environ["H3_VAE_PRECISION"] = mode
                got = torch.empty(truth.shape, dtype=torch.float32)
                video = latent.contiguous().float()
                started = time.monotonic()
                ok = library.h3_phase1_video(native_weights.encode(), video.data_ptr(),
                                             *video.shape[2:], got.data_ptr(), got.numel(),
                                             shape, error, len(error))
                if not ok:
                    raise RuntimeError(f"{name} {label}: {error.value.decode()}")
                if tuple(shape) != tuple(got.shape[:3]):
                    raise ValueError(f"{name} {label}: native shape differs")
                results[label] = (got, time.monotonic() - started)
            for label, (got, seconds) in results.items():
                row = dict(case=name, decoder=label, frames=int(truth.shape[0]),
                           height=int(truth.shape[1]), width=int(truth.shape[2]),
                           seconds=seconds, **metrics(got, truth))
                rows.append(row)
                print(json.dumps(row), flush=True)
            del results
            gc.collect()
            torch.cuda.empty_cache()
    verdicts = []
    for name, _ in cases:
        by = {r["decoder"]: r for r in rows if r["case"] == name}
        ours, bound = by["h3-fp16"], by["torch-fp16"]
        verdicts.append(dict(
            case=name, passed=ours["finite"] and
            ours["relative_l2"] <= bound["relative_l2"] and
            ours["normalized_max"] <= bound["normalized_max"],
            h3_fp16_relative_l2=ours["relative_l2"], torch_fp16_relative_l2=bound["relative_l2"],
            h3_fp16_normalized_max=ours["normalized_max"],
            torch_fp16_normalized_max=bound["normalized_max"]))
    report = dict(scope="Video VAE decode precision against PyTorch FP32 on identical latents",
                  acceptance="h3-fp16 finite and no worse than torch-fp16 (diffusers FP16) "
                             "in relative L2 and normalized max, every case",
                  passed=all(v["passed"] for v in verdicts), verdicts=verdicts, rows=rows)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], verdicts=verdicts), indent=1))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
