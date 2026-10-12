# Finite live serving acceptance — job 15243

Date: 2026-10-12. Runtime source: `f45f0da`. Two H200s on `gpu-h200x8-0`,
Apptainer `/data0/apptainer/pytorch_291_cu128.sif`, native TaoMate weights at
`/data0/models/h3-taomate/native`. The resident pilot remains running on port
18961. FastH3's separate job and endpoint are unchanged.

The browser sent one prompt and received twelve connected requests over a
single fragmented MP4 response. History was committed 48 times; the native
timeline reached 1,433 frames, 422 video latents and 2,388 audio latents.
The worker reported 173.786 seconds for the complete session. The browser
played media after 20.789 seconds while generation was still running, received
the full stream after 173.985 seconds, and downloaded a 22,689,190-byte file.
This is functional finite streaming, not sustained real-time performance.

The sample is available on the development host at
`/home/homo/model/autograd-h3-live/run/h3-live/live-12-parts.mp4`.
SHA-256: `59ea947f808af859b68cf8d8feb9fdd08c76b9317caf6837594981354afeaac9`.
FFmpeg 7.0.2 decoded all 1,433 frames without errors. The native video duration
is 59.7083 seconds; Chrome reports 59.7404 seconds including codec timing.
The browser had no media or JavaScript errors. Download, mobile layout and
Stop passed. The service logged cancellation and subsequently accepted a
portrait session, delivering its first 65,536 media bytes before that test
also disconnected and cancelled.

Adjacent frames around all eleven request boundaries were visually inspected.
No obvious identity or background switch was seen. This check covers one
landscape prompt with a random seed, not general continuity or speech quality.
The browser test used a loopback Auth0 fixture with the real worker. Public
deployment checks separately covered service health, assets, 401 and CORS.

The first run exposed a final audio-length check failure: 2,388 audio latents
decode to 1,910,400 samples, 267 samples short of the rounded video endpoint.
The adapter now retains the native final audio length. The corrected full
browser run passed; `tests/live/stream_lengths.c` checks all finite endpoints
and cancellation under ASan/UBSan. The fragmented encoder CPU test and all
three HTTP admission/stream/cancellation tests also passed.

Remote operational evidence:

- `/home/cj/h3-live-service/logs/service-15243.log`
- `/home/cj/h3-live-service/logs/worker-15243.log`
- `/home/cj/h3-live-service/source-15243.sha256` (initial build inputs)
- `/home/cj/h3-live-service/endpoint-fix.sha256` (committed runtime fix)

The startup overlay was also updated to `f45f0da` so a later restart retains
the fix. Autograd's committed `deploy/h3-live/acceptance-20261012.json` records
the image, checks and browser results. Its `apps/web/tests/live-video.py` is
the browser acceptance driver. The two-GPU pilot uses the bounded prefix
decoder described in `cuda/LIVE.md`.
