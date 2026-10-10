"""Prepare the FlashAttention sources used by the CUDA backend.

flash-cuda is the revision bundled with the pinned PyTorch oracle; only its
dropout-free BF16/128 forward kernel is compiled, with the unused ATen RNG
plumbing removed. flash3-hopper is the FlashAttention-3 Hopper forward used
for non-causal DiT attention; its headers are used unchanged.
Run before requesting a GPU.
"""
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
SOURCES = (
    ("Dao-AILab/flash-attention", "979702c87a8713a8e0a5e9fee122b90d2ef13be5",
     "ea1ec2e43422a2b910ecad9df7d4fd58f09ff39036e03d037888147f784ca407", "flash-cuda"),
    ("NVIDIA/cutlass", "e51efbfe18fe4f4cbb66ab814c55bf4aa0185491",
     "1fc686a38edacbb5b7416c2835efb6801d2a583a7942bb5b78ac6f4999cd73a1", "cutlass-flash"),
    # FlashAttention-3 Hopper forward (flash-attention v2.8.3) and its CUTLASS.
    ("Dao-AILab/flash-attention", "060c9188beec3a8b62b33a3bfa6d5d2d44975fab",
     "64391cfec73fc5a25932b2c0c2fb3b4d9f97af001164e5220af21049fc17e04e", "flash3-hopper"),
    ("NVIDIA/cutlass", "dc4817921edda44a549197ff3a9dcf5df0636e7b",
     "f2a3a9df5e6f010c8b02716aa2644a6f071827fafa606fac5f5241cab6a1ab56", "cutlass-flash3"),
)


def main():
    for repository, revision, digest, folder in SOURCES:
        destination = DEPS / folder
        marker = destination / "h3-revision"
        if marker.exists() and marker.read_text().strip() == revision:
            continue
        archive = urllib.request.urlopen(f"https://codeload.github.com/{repository}/tar.gz/{revision}").read()
        if hashlib.sha256(archive).hexdigest() != digest:
            raise ValueError(repository + ": archive digest differs")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                relative = Path(*Path(member.name).parts[1:])
                if ".." in relative.parts:
                    raise ValueError("Unsafe archive path")
                if folder == "flash3-hopper":
                    hopper = relative.parent == Path("hopper") and relative.suffix in (".h", ".hpp")
                    if not hopper and (relative.name != "LICENSE" or len(relative.parts) != 1):
                        continue
                    if hopper:
                        relative = Path(relative.name)
                elif repository.endswith("flash-attention"):
                    prefix = Path("csrc/flash_attn/src")
                    if relative.is_relative_to(prefix) and relative.suffix in (".h", ".cuh"):
                        relative = relative.relative_to(prefix)
                    elif relative.name != "LICENSE" or len(relative.parts) != 1:
                        continue
                elif not relative.is_relative_to("include") and relative.name != "LICENSE.txt":
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(member).read())
        if folder == "flash-cuda":
            substitutions = {
                "flash.h": {
                    "#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState":
                        "#include <cstdint> // H3: dropout-free standalone forward, no ATen runtime",
                    "at::PhiloxCudaState philox_args;": "uint64_t unused_philox_args[4];",
                },
                "flash_fwd_kernel.h": {
                    '#include "philox_unpack.cuh" // For at::cuda::philox::unpack':
                        "#include <tuple> // H3: dropout is disabled at template instantiation",
                    "auto seed_offset = at::cuda::philox::unpack(params.philox_args);":
                        "auto seed_offset = std::make_tuple(uint64_t(0), uint64_t(0));",
                },
            }
            for name, changes in substitutions.items():
                path = destination / name
                text = path.read_text()
                for before, after in changes.items():
                    if text.count(before) != 1:
                        raise ValueError("Unexpected upstream text: " + name)
                    text = text.replace(before, after)
                path.write_text(text)
        marker.write_text(revision + "\n")
        print("Prepared", repository, revision, flush=True)


if __name__ == "__main__":
    main()
