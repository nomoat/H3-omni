"""Compare native video/audio decoders with the original PyTorch modules."""
import argparse
import ctypes as C
import gc
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
from real_dit import conversion_module


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root, output = Path(args.weights), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latents = load_file(args.latents)
    video, audio = latents["video"], latents["audio"]
    conversion = conversion_module(args.diffusers)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    library = C.CDLL(str(Path("build-cuda/libh3_model.so").resolve()))
    pointer, size, integer = C.c_void_p, C.c_size_t, C.c_int
    library.h3_phase1_video.argtypes = [C.c_char_p, pointer, integer, integer, integer, pointer, size, pointer, pointer, size]
    library.h3_phase1_video.restype = integer
    library.h3_phase1_audio.argtypes = [C.c_char_p, pointer, integer, pointer, size, pointer, pointer, size]
    library.h3_phase1_audio.restype = integer
    error = C.create_string_buffer(1024)
    shape = (integer * 3)()
    limits = json.loads(Path(__file__).with_name("acceptance.json").read_text())["model"]
    records = []

    def compare(label, got, expected, seconds, scope="identical_latents"):
        expected = expected.detach().cpu().float().contiguous()
        if got.shape != expected.shape:
            raise ValueError(f"{label}: native shape {got.shape}, reference {expected.shape}")
        delta = got - expected
        rel = float(delta.norm() / expected.norm().clamp_min(1e-12))
        maximum = float(delta.abs().max() / expected.abs().max().clamp_min(1e-12))
        threshold = limits["decoded_" + label + "_float"]
        passed = bool(torch.isfinite(got).all() and torch.isfinite(expected).all() and
                      rel <= threshold["relative_l2"] and maximum <= threshold["normalized_max"])
        row = dict(component=label, scope=scope, passed=passed, relative_l2=rel, normalized_max=maximum,
                   native_load_and_decode_seconds=seconds)
        records.append(row)
        print(json.dumps(row), flush=True)
        save_file({"native": got, "reference": expected}, str(output / f"{label}-{scope}.safetensors"))

    with torch.inference_mode():
        # Float32 checkpoint; do not introduce BF16 decoder weight conversion.
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
        del state
        mean = torch.tensor(config["latents_mean"], device="cuda").view(1, 24, 1, 1, 1)
        std = torch.tensor(config["latents_std"], device="cuda").view(1, 24, 1, 1, 1)
        decoded = model.decode(video.cuda() * std + mean).sample
        pixel_mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1, 1)
        expected = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1).squeeze(0).permute(1, 2, 3, 0).cpu()
        trajectory_decoded = model.decode(latents["reference_video"].cuda() * std + mean).sample
        trajectory_expected = (trajectory_decoded.float() * pixel_std + pixel_mean).clamp(0, 1).squeeze(0).permute(1, 2, 3, 0).cpu()
        del trajectory_decoded
        del model, decoded
        gc.collect()
        torch.cuda.empty_cache()
        got = torch.empty(expected.shape, dtype=torch.float32)
        started = time.monotonic()
        ok = library.h3_phase1_video(str(root / "video_vae" / wrapper["source_path"]).encode(),
            video.data_ptr(), *video.shape[2:], got.data_ptr(), got.numel(), shape, error, len(error))
        if not ok:
            raise RuntimeError(error.value.decode())
        if tuple(shape) != tuple(got.shape[:3]):
            raise ValueError("Native video metadata differs")
        compare("video", got, expected, time.monotonic() - started)
        compare("video", got, trajectory_expected, time.monotonic() - started, "independent_trajectory")

        config = conversion.get_audio_vae_config(str(root))
        # The small audio model can initialize normally, preserving derived buffers.
        model = AutoencoderKLMiniMaxH3Audio(**config)
        model.load_state_dict(load_file(str(root / "audio_vae/model.safetensors")), strict=True)
        model = model.cuda().eval()
        mean = torch.tensor(config["latents_mean"], device="cuda").view(1, 32, 1)
        std = torch.tensor(config["latents_std"], device="cuda").view(1, 32, 1)
        decoder_input = audio.squeeze(0).permute(1, 0, 2).cuda()
        expected = model.decode(decoder_input * std + mean).sample.squeeze(1).cpu()
        reference_decoder_input = latents["reference_audio"].squeeze(0).permute(1, 0, 2).cuda()
        trajectory_expected = model.decode(reference_decoder_input * std + mean).sample.squeeze(1).cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        got = torch.empty(expected.shape, dtype=torch.float32)
        started = time.monotonic()
        ok = library.h3_phase1_audio(str(root / "audio_vae").encode(), audio.data_ptr(), audio.shape[-1],
                                    got.data_ptr(), got.numel(), shape, error, len(error))
        if not ok:
            raise RuntimeError(error.value.decode())
        if tuple(shape) != (*got.shape, 32000):
            raise ValueError("Native audio metadata differs")
        compare("audio", got, expected, time.monotonic() - started)
        compare("audio", got, trajectory_expected, time.monotonic() - started, "independent_trajectory")
    report = dict(scope="Decoder parity on identical latents and independent trajectories",
                  latents=str(Path(args.latents).resolve()),
                  passed=all(r["passed"] for r in records), cases=records)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
