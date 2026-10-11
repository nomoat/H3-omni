# FastH3 Preview v1 on H200

The optional CUDA backend runs the four-step FastH3 Preview v1 VSA checkpoint
as a standalone C/CUDA executable. Python and PyTorch are used only for weight
conversion and independent validation. The executable does not link libtorch.

This path is for text-to-video/audio on one Hopper GPU. It preserves all 50
transformer blocks. Image anchors, Ref2VA, arbitrary step counts, layer skipping,
quantization and reuse are rejected for this checkpoint.

## Prepare and run

Prepare dependencies on the login node, then build inside the documented CUDA
container on an allocated H200. See the workspace `GPU-ACCESS.md`.

```sh
python3 cuda/prepare_flash.py
python3 cuda/prepare_vsa.py
make -f cuda/Makefile -j8 VSA=1 CUDA_ARCH=sm_90
```

The Hopper kernel requires C++20 and CUDA 12.8. Other translation units retain
their existing C11/C++17 settings. `VSA=0` remains the default build. Changing
`VSA` rebuilds the affected objects.

Download the full checkpoint (not the small LoRA adapter):

```sh
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --revision 5ea076f35b84da4c3c82217112fa733d8eea2ae1 \
  --include 'transformer/*' 'scheduler/*' 'audio_scheduler/*' \
            'fastvideo_inference.json' 'README.md' 'LICENSE' \
  --local-dir /path/to/fasth3-source
python3 cuda/prepare_fasth3.py \
  --source /path/to/fasth3-source --base /path/to/MiniMax-H3 \
  --output /path/to/fasth3-native
```

Conversion requires CPU PyTorch and safetensors. It writes a new model directory
and symlinks the original tokenizer, text encoder and VAEs. Keep those base
components available. It refuses to overwrite an existing output.

```sh
build-cuda/h3 -d /path/to/fasth3-native \
  -p 'A cat walks through a sunny garden.' \
  --width 864 --height 480 --seconds 5 --seed 0 --profile -o cat.mp4
```

The manifest selects four model evaluations automatically. `--steps 4` is also
accepted. C API callers should initialize `params.steps` with
`h3_model_default_steps(ctx)`. Original H3 checkpoints still default to 20.

## Model contract

- DMD rungs `[999, 749, 500, 250]`, followed by zero. Each rung is divided by
  1000 and shifted once: video shift 12, audio shift 3. These differ from
  original H3's uniform `--steps 4` schedule.
- VSA uses 64-token tiles. Video tiles are 4x4x4 in the patchified token grid;
  text and audio have separate dense tiles. Padding never contributes to means
  or attention. Video queries select `ceil(video_tiles / 10)` video tiles per
  head, and always retain text/audio keys. Text/audio queries stay dense.
- FP32 pooled Q/K scores select the blocks and drive a pooled compression
  attention. All 50 learned compression gates are preserved. The gate has no
  sigmoid. Compression is rounded to BF16, then combined with the gate and
  sparse output by a fused multiply-add with BF16 output.
  Pooling and softmax preserve PyTorch 2.9.1's reduction order, including the
  softmax kernel switch above 2048 tiles.
- The two text-refiner layers use FlashAttention 2 even with `H3_FLASH3=1`.
  This preserves the reference's rounding. The core blocks use VSA.
- Conversion interleaves Q/K/V per head and reverses the two SwiGLU halves.
  Input, timestep and final projections are widened exactly from BF16 to FP32,
  matching FastVideo's execution dtype policy. No quantization is introduced.
  The derived `rope.inv_freq` buffer, omitted by Diffusers, is regenerated from
  the checkpoint's RoPE configuration.
- The runtime rejects a malformed manifest or a VSA gate/manifest mismatch.
  A build without `VSA=1` rejects VSA weights. Maximum geometry is 4096 tiles.

The exporter also recognizes Preview v1's full dense checkpoint contract. The
real-checkpoint validation described here targets the VSA release; it does not
establish separate acceptance for the dense release or LoRA-only checkpoints.

## Independent checks

```sh
python3 tests/fasth3/prepare_reference.py
python3 tests/fasth3/vsa.py --fastvideo .deps/fasth3-reference --output vsa.json
python3 tests/fasth3/contract.py --output contract.json
python3 tests/fasth3/parity.py \
  --source /path/to/fasth3-source --fastvideo .deps/fasth3-reference --oracle hopper \
  --weights /path/to/fasth3-native/FL2VA --diffusers /path/to/pinned-diffusers \
  --text /path/to/real-text.safetensors \
  --width 256 --height 256 --frames 22 --seed 0 --output parity
```

Validation has two levels. The sparse kernel is checked against independent
PyTorch FP32 math SDPA in `vsa.py`. The full-model integration reference loads
unconverted weights into pinned Diffusers. It uses unchanged FastVideo
tile/mask helpers and PyTorch pooling, block selection, compression and gates.
Only the unmodified upstream Hopper sparse kernel is shared with the native
runtime. Both identical-input velocities and independent four-step trajectories
use the existing model thresholds. This is not a run of the full FastVideo
application or a reproduction of its B200 measurements.

`--oracle math` is a precision-sensitivity diagnostic. It replaces BF16 Hopper
attention with FP32 math attention. That comparison **does not pass** the
full-model thresholds: this checkpoint amplifies small attention differences
through its 50 layers. The recorded 256x256, 22-frame case has final latent
relative L2 differences of 0.518 (video) and 1.115 (audio). In the same-kernel
Hopper comparison, all 16 velocity/latent comparisons are bit-identical. Do not
interpret the two attention implementations as interchangeable for this model.

The FA3 text-refiner diagnostic also fails: before pinning those two layers to
FA2, the 512x512 Chinese-prompt case ended with latent relative L2 differences
of 0.637 (video) and 0.215 (audio). The corrected default configuration has
bit-identical four-step latents and velocities for that case. The failed
diagnostics are retained alongside the passing reports.

Large geometry also requires matching reduction order. A sequential tile sum
and a different softmax sum produced only ten differing BF16 attention elements
out of 110 million in the first 864x480, 124-frame block, but the final video
latent relative L2 reached 0.595. Matching the reference's FP32 addition order
removes those differences. The corrected complete four-step run is bit-identical.

## Recorded validation

Pinned sources and environment are in
[`tests/fasth3/sources.json`](../tests/fasth3/sources.json). Compact reports are
under [`tests/fasth3/evidence`](../tests/fasth3/evidence); full-model parity tensors for the accepted runs are
on the cluster at `/home/cj/h3-fasth3/results/<job>`.

| Check | Result | Evidence |
| --- | --- | --- |
| Export schema | All 585 native tensors, including 50 learned gates, match the required shapes and dtypes | `export/schema.json` |
| DMD schedule, Euler update, manifest rejection and derived RoPE | 16 checks pass | `14583/contract.json` |
| VSA vs independent FP32 math | 8 cases pass; largest relative L2 is 0.00234217 | `14583/vsa-math.json` |
| VSA orchestration vs upstream Hopper kernel | 12 cases are bit-identical, including 288 and 2051 tiles | `14583/vsa-hopper.json` |
| Four-step model, 256x256, 22 frames, English, seed 0 | All 16 velocity/latent comparisons are bit-identical | `14569/parity/results.json` |
| Four-step model, 512x512, 22 frames, Chinese, seed 1, default settings | All 16 velocity/latent comparisons are bit-identical | `14576/parity/results.json` |
| Four-step model, 864x480, 124 frames, English, seed 43, default settings | All 16 velocity/latent comparisons are bit-identical after matching reduction order | `14583/fullsize.json` |
| Video/audio decoders, identical inputs and independent trajectories | All 8 comparisons pass across reference and default settings | `14569/decoded/results.json`, `14576/decoded/results.json` |
| Original H3, 20 evaluations | All 80 velocity/latent comparisons are bit-identical | `14535/original.json` |
| CUDA build with and without VSA | Build switching, 69 existing primitives and VSA rejection without support pass | `14538/` |
| Existing CUDA primitives after the reduction-order fix | All 69 checks pass | `14583/primitives.json` |
| Standalone CLI and resident rebind | Automatic four steps, MP4/audio output and invalid-settings rejection pass; repeated request is byte-identical after changing geometry and prompt | `14577/smoke/results.json` |

## H200 measurement and prepared files

Job 14583 generated 864x480, 124 frames at 24 fps (5.1667 seconds), with audio,
all 50 blocks, four evaluations and no tensor dumps. The prompt was
`A cat walks through a sunny garden.` The same resident process ran seed 42,
then seed 43. The second request reused weights and the prompt cache.

| Request | Wall time |
| --- | ---: |
| First request, including model loading from shared storage | 258.93 s |
| Second request, weights and prompt cached, including decoding and MP4 output | 22.53 s |
| Four-step denoising portion of the second request | 5.13 s |

After the merge with the FP16 video VAE (job 14779, gpu-h200x1-2, resident
`h3_serve_check`), warm 864x480 requests take 10.5-11.6 s for 5 s video and
21.0-22.5 s for 10 s video: denoise 5.1 / 10.8 s, video VAE 2.5 / 4.9 s,
and 2.4-6.3 s of MP4 encoding that does not overlap the decode.

Multi-GPU (`--gpus N`, job 14963, 2 GPUs, gpu-h200x8-1): VSA attention runs
with 28 heads per GPU after the Ulysses exchange, and the gate is exchanged
like Q/K/V. Latents, audio and RGB are bit-identical to one GPU on two
requests. Warm 864x480 requests: 5 s video 4.55-4.69 s (denoise 2.95 s,
video VAE 1.18 s, MP4 tail 0.18 s), 10 s video 8.91-9.02 s. On this 8-GPU
node the MP4 tail is 0.2 s; the 2.4-6.3 s tail of job 14779 came from the
single-GPU node.

| Warm request, 864x480 | 1 GPU (14779) | 2 GPUs (14963) | 4 GPUs (14961) | 8 GPUs (14962) |
| --- | --- | --- | --- | --- |
| 5 s video | 10.5-11.6 s | 4.55-4.69 s | 2.62-2.81 s | 1.85-2.21 s |
| 10 s video | 21.0-22.5 s | 8.91-9.02 s | 5.07-5.18 s | 3.47-3.84 s |

All multi-GPU runs are bit-identical to one GPU on the two check requests.
At 8 GPUs a 5 s request spends 0.85-0.9 s in denoise and 0.6 s in the video
VAE, whose host passes no longer shrink with more GPUs.

These are individual requests, not a statistical benchmark or a comparison
with the blog's B200 system. Cold loading depends heavily on shared storage.
Raw timings are in `14583/benchmark.jsonl`. The actual warm MP4 has
124 frames and 32 kHz audio; metadata is in `14583/warm-media.json`.

Prepared paths on the documented cluster:

- Native model: `/data0/models/h3-fasth3/vsa-native`.
- Unconverted source: `/data0/models/h3-fasth3/vsa-source`.
- Final executable: `/home/cj/h3-fasth3/results/14587/h3`.
- Example MP4: `/home/cj/h3-fasth3/results/14583/benchmark/warm/output.mp4`.

Run inside the documented CUDA container on an allocated Hopper GPU. The
executable needs the existing ICU runtime and FFmpeg. For this cluster:

```sh
export LD_LIBRARY_PATH=/var/tmp/h3-fasth3/icu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export H3_FFMPEG=/home/cj/h3-cuda/releases/14274/bin/ffmpeg
/home/cj/h3-fasth3/results/14587/h3 -d /data0/models/h3-fasth3/vsa-native \
  -p 'A cat walks through a sunny garden.' \
  --width 864 --height 480 --frames 124 --seed 43 --profile -o cat.mp4
```

The ICU staging directory above is on `gpu-h200x1-2`; on another node, unpack
`/home/cj/h3-cuda/icu.tar` into a local directory and point `LD_LIBRARY_PATH` to
its `usr/lib/x86_64-linux-gnu`. The final rebuild only changes CLI help text;
the model library hash is identical to job 14583. Its linked dependencies are
recorded in `14587/ldd.txt`, with no Python or libtorch dependency.

Upstream: [FastH3 preview](https://haoailab.com/blogs/fasth3-preview/),
[checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/tree/5ea076f35b84da4c3c82217112fa733d8eea2ae1),
[VSA source](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/video_sparse_attn_h3.py).
Third-party code and licenses are listed in [VSA-NOTICE.md](vendor/VSA-NOTICE.md).
