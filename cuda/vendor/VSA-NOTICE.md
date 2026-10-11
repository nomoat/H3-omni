The optional FastH3 VSA backend includes third-party code:

- `vsa_hopper_forward.cuh` contains the forward kernel from
  [FastVideo](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo-kernel/csrc/attention/block_sparse_h100.cu),
  revision `d5253fd71055927f12aa6c4aa679b93a00238851`, under Apache-2.0.
  The license is retained as `FASTVIDEO-LICENSE`. Changes remove the PyTorch
  include, backward kernels and Python bindings; forward arithmetic is retained.
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens/tree/6c27e28c8115d1839d9eeeb530913c184a75fc87)
  headers are fetched by `../prepare_vsa.py` at revision
  `6c27e28c8115d1839d9eeeb530913c184a75fc87`. Its license is retained in
  `.deps/thunderkittens/LICENSE`.

The native orchestration and C ABI are in `../vsa.cuh` and `../vsa_hopper.cu`.
FastH3 model weights are obtained separately and retain their upstream licenses.
