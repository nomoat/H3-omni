"""Kernel profile of the resident tiled video-VAE decoder (the serving path).
Decodes a random normalized latent once to warm up, then once under
torch.profiler, which records every CUDA kernel in the process."""
import argparse
import ctypes as C
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--time", type=int, default=67)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--width", type=int, default=54)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.weights)
    wrapper = json.loads((root / "video_vae/config.json").read_text())
    weights = str(root / "video_vae" / wrapper["source_path"]).encode()
    lib = C.CDLL(str(Path("build-cuda/libh3_model.so").resolve()))
    p, i, s = C.c_void_p, C.c_int, C.c_size_t
    lib.h3_video_vae_decoder_load.restype = p
    lib.h3_video_vae_decoder_load.argtypes = [C.c_char_p, C.c_char_p, i, i, p, p, p, s]
    lib.h3_video_vae_decoder_decode.argtypes = [p, p, i, p, p, s]
    lib.h3_video_vae_decoder_decode.restype = i
    lib.h3_video_frames_free.argtypes = [p]
    error = C.create_string_buffer(1024)
    decoder = lib.h3_video_vae_decoder_load(weights, b"h3_shaders.metal", args.height, args.width,
                                            None, None, error, len(error))
    if not decoder:
        raise RuntimeError(error.value.decode())
    torch.manual_seed(0)
    latent = torch.randn(24, args.time, args.height, args.width).contiguous()
    frames = (C.c_char * 256)()

    def decode():
        started = time.monotonic()
        if not lib.h3_video_vae_decoder_decode(decoder, latent.data_ptr(), args.time, frames,
                                               error, len(error)):
            raise RuntimeError(error.value.decode())
        seconds = time.monotonic() - started
        lib.h3_video_frames_free(frames)
        return seconds

    print("warmup", decode(), flush=True)
    print("timed", decode(), flush=True)
    ffmpeg = os.environ.get("H3_FFMPEG")
    if ffmpeg:
        # The serving path encodes MP4 with libx264 while the decode runs;
        # load the CPUs the same way to see whether the decode slows down.
        encoder = subprocess.Popen([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                                    "testsrc2=s=864x480:r=24", "-t", "60", "-c:v", "libx264",
                                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                                    "-f", "null", "-"])
        time.sleep(1)
        print("timed with x264 running", decode(), flush=True)
        encoder.kill()
        encoder.wait()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        print("profiled", decode(), flush=True)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=90)
    print(table)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "kernels.txt").write_text(table)
    prof.export_chrome_trace(str(Path(args.output) / "trace.json"))


if __name__ == "__main__":
    main()
