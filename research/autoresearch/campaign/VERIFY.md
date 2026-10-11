The evaluator owns `verify.py`, the reference cache and all files in
`tests/pytorch`. Candidate code cannot change these files or their thresholds.
All commands print one final `PASS` or `FAIL`, return exit code 0 or 1, and write
`verification.json`. The output directory must be new or empty. Stage logs and
numeric comparisons remain in that directory. These wall times are correctness
costs, not performance scores.

Capture primitive and host reference fixtures once inside the allocated H200
job and pinned Python environment:

```sh
python3 research/autoresearch/campaign/frozen_primitives.py capture \
  --tests-dir "$H3_FROZEN_SEED/tests/pytorch" --output-dir "$H3_UNIT_ORACLE_DIR"
```

Prepare the full cache once (this step uses CPU only and copies the captured
unit fixtures):

```sh
python3 research/autoresearch/campaign/verify.py prepare \
  --reference-root /home/cj/h3-cuda/results/14274 \
  --diffusers "$H3_PINNED_DIFFUSERS" --unit-oracle "$H3_UNIT_ORACLE_DIR" \
  --output "$H3_ORACLE_DIR"
```

Read and freeze `manifest_sha256` from `verification.json`. Pass that exact value
in subsequent commands. The manifest hashes every cached tensor, the source
artifacts and the fixed judge files. The cache can be copied to local storage;
its hash must stay unchanged. Changing the evaluator requires explicit review
and a new cache preparation; a candidate cannot update the frozen hash.

Run each candidate's screen inside its allocated H200 job:

```sh
python3 research/autoresearch/campaign/verify.py screen \
  --oracle "$H3_ORACLE_DIR" --oracle-sha256 "$H3_FROZEN_ORACLE_SHA256" \
  --candidate-root "$H3_CANDIDATE_DIR" --weights "$H3_FL2VA_DIR" \
  --diffusers "$H3_PINNED_DIFFUSERS" --output "$H3_ROUND_SCREEN_DIR"
```

The screen includes 69 primitive cases, 32 layout cases, 12 scheduler/sampler
cases, tokenizer checks, the 82-token text embedding, 20 independent denoising
steps at 864×480×56, velocity checks on each reference input, and video/audio
float output checks. It uses the original precision limits. No reference
velocity advances the candidate trajectory. A candidate forward also supplies
the velocity check only when its text and both input latents are bitwise equal
to the oracle inputs. Otherwise, an additional forward uses the frozen oracle
input. This avoids redundant execution for exact candidates while retaining
every comparison. Decoder results can likewise be reused across the two checks
only when both decoder input latents are bitwise identical.

The cache covers all six regression trajectories, the extended trajectory and
three text embeddings. It is valid because job 14274 text and all trajectory
comparisons were exact. Preparation checks those records, verifies the saved
native/reference velocities, reconstructs reference states using the pinned
official FP32 scheduler, and requires an exact match to each saved independent
CUDA reference's final state. Any discrepancy fails preparation. Earlier job
14269 artifacts cannot be used.

A screening PASS is not acceptance. Before promotion run:

```sh
python3 research/autoresearch/campaign/verify.py promote \
  --oracle "$H3_ORACLE_DIR" --oracle-sha256 "$H3_FROZEN_ORACLE_SHA256" \
  --candidate-root "$H3_CANDIDATE_DIR" --weights "$H3_FL2VA_DIR" \
  --diffusers "$H3_PINNED_DIFFUSERS" --output "$H3_PROMOTION_DIR"
```

Promotion runs all six regression cases and the extended case against this
fixed cache, plus three text embeddings, primitive/host checks, float decoders
and a standalone CLI media check. No large PyTorch reference model runs during
screening or promotion. Primitive, host and tokenizer outputs are also frozen.
Candidate unit tests replay the captured inputs and compare against those fixed
outputs; they do not recalculate a PyTorch reference. CPU tensor operations in
the evaluator only arrange candidate inputs and measure errors.

This changes the velocity diagnostic's input policy, not its numeric limits:
velocity is compared at fixed oracle states, instead of evaluating a new
PyTorch velocity on each candidate's current state. The candidate's independently
evolving state is still compared after every step, and its final decoded output
is compared with the fixed independent PyTorch result. Neither gate claims the
old live candidate-state velocity comparison. The accepted bridge covers the current campaign's
changes in CUDA scratch allocation, descriptors, GEMM parameters and launch
geometry. Production orchestration changes and a new device sampler need an
additional reviewed production-path numerical gate.

Candidate builds must contain `build-cuda/h3`, `libh3_cuda.so`, `libh3_model.so`
and `libh3_tokenizer.so` from the same source/configuration. Their hashes are
recorded. The runner must enforce an overall process timeout in addition to the
per-subprocess `--stage-timeout` (default 3600 seconds); ctypes/native calls can
only be stopped safely by terminating the worker process.

Local CPU-only judge tests:

```sh
python3 research/autoresearch/campaign/test_verify.py
```
