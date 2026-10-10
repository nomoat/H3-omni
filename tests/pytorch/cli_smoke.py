"""Run the standalone executable, then decode its actual video/audio file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = Path("build-cuda/h3").resolve()
    movie = args.output / "cli-smoke.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    environment = {**os.environ, "H3_FFMPEG": ffmpeg}
    command = [str(binary), "-d", str(Path(args.weights).resolve().parent), "-p", "A cat walks through a sunny garden.",
               "--width", "256", "--height", "256", "--frames", "22", "--steps", "20",
               "--layers", "50", "--reuse", "1", "--core-reuse", "1", "--seed", "0",
               "--profile", "-o", str(movie)]
    started = time.monotonic()
    with (args.output / "cli-smoke.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.monotonic() - started
    reader = imageio_ffmpeg.read_frames(str(movie))
    metadata = next(reader)
    frames = sum(1 for frame in reader)
    audio = subprocess.run([ffmpeg, "-v", "error", "-i", str(movie), "-map", "0:a:0",
                            "-f", "f32le", "-acodec", "pcm_f32le", "-"],
                           check=True, capture_output=True).stdout
    passed = frames == 22 and tuple(metadata["size"]) == (256, 256) and len(audio) > 0
    result = dict(passed=passed, command=command, frames=frames, steps=20, layers=50,
                  video_size=metadata["size"], decoded_audio_bytes=len(audio), elapsed_seconds=elapsed,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  movie_sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    (args.output / "cli-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
