"""Native VSA against independent math or upstream-kernel integration oracles.

The default math oracle checks the sparse kernel independently. The Hopper
oracle demands exact orchestration, including large tile counts and reduction
boundaries. Both use unchanged upstream tile/mask helpers.
"""
import argparse
import ast
import ctypes as C
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pytorch'))
from native import Native


def upstream_helpers(checkout):
    namespace = dict(torch=torch, math=math, _TILE_ELEMS=64, VSA_H3_TILE_SIZE=(4, 4, 4),
                     VSA_TILE_SIZE=(4, 4, 4))
    paths = {
        'video_sparse_attn.py': ['get_tile_partition_indices', 'construct_variable_block_sizes',
                                  'get_non_pad_index', 'compute_topk'],
        'video_sparse_attn_h3.py': ['_validate_h3_segment_geometry', '_h3_segment_tile_geometry',
                                     '_pool_tiles', '_build_block_mask'],
    }
    for filename, names in paths.items():
        source = checkout / 'fastvideo/attention/backends' / filename
        tree = ast.parse(source.read_text())
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in definitions} == set(names)
        for node in definitions:
            node.decorator_list = []
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future] + definitions, type_ignores=[]))
        exec(compile(module, str(source), 'exec'), namespace)
    return namespace


def hopper_attention(q, k, v, mask, sizes, library):
    """Raw upstream Hopper kernel; pooling, selection and merge remain PyTorch.

    Used to isolate integration from BF16 probability rounding. The independent
    math-SDPA primitive checks remain the validation of this shared kernel.
    """
    fn = library.h3_vsa_hopper
    fn.argtypes = [C.c_void_p] * 8 + [C.c_uint, C.c_uint, C.c_void_p]
    fn.restype = C.c_int
    q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    tiles, heads = mask.shape[-1], q.shape[1]
    candidates = torch.arange(tiles, dtype=torch.int32, device=q.device).expand(mask.shape)
    indices = torch.where(mask, candidates, tiles).sort(-1).values.contiguous()
    counts = mask.sum(-1, dtype=torch.int32).contiguous()
    sizes = sizes.to(torch.int32).contiguous()
    out, lse = torch.empty_like(q), torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    code = fn(*[x.data_ptr() for x in (out, q, k, v, lse, indices, counts, sizes)],
              tiles, heads, torch.cuda.current_stream().cuda_stream)
    if code:
        raise RuntimeError(f'upstream Hopper kernel returned CUDA error {code}')
    return out.transpose(1, 2)


def reference(q, k, v, gate, text, audio, geometry, upstream, hopper=None):
    parts, sizes, untile, prefix, videos, spans = upstream['_h3_segment_tile_geometry'](
        (text, audio, geometry), q.device, (4, 4, 4))
    nonpad = upstream['get_non_pad_index'](sizes, 64)
    buffers = []
    for x in (q, k, v):
        dst = torch.zeros((1, sizes.numel() * 64, *x.shape[-2:]), dtype=x.dtype, device=x.device)
        dst[:, nonpad] = x[:, parts]
        buffers.append(dst)
    qt, kt, vt = buffers
    pool = lambda x: upstream['_pool_tiles'](x, sizes, 64)
    qp, kp, vp = map(pool, buffers)
    scores = qp @ kp.transpose(-2, -1) / math.sqrt(128)
    mask = upstream['_build_block_mask'](scores, prefix, .9, True, spans, (.9,))
    if hopper is None:
        dense_mask = mask.repeat_interleave(64, -2).repeat_interleave(64, -1)
        valid = torch.arange(qt.shape[1], device=q.device) % 64 < sizes.repeat_interleave(64)
        dense_mask &= valid[None, None, None, :]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sparse = F.scaled_dot_product_attention(qt.transpose(1, 2).float(), kt.transpose(1, 2).float(),
                                                    vt.transpose(1, 2).float(), attn_mask=dense_mask)
        sparse = sparse.transpose(1, 2).bfloat16()
    else:
        sparse = hopper_attention(qt, kt, vt, mask, sizes, hopper)
    sparse = sparse[:, untile]
    compress = (scores.softmax(-1) @ vp).transpose(1, 2).bfloat16().repeat_interleave(64, 1)[:, untile]
    return torch.addcmul(sparse, compress, gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastvideo', type=Path, required=True)
    parser.add_argument('--library', default='build-cuda/libh3_cuda.so')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle', choices=['math', 'hopper'], default='math')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1234)
    upstream = upstream_helpers(args.fastvideo)
    native = Native(args.library)
    records = []
    try:
        cases = [(3, 11, (1, 2, 3), 2), (65, 67, (5, 7, 9), 4),
                 (17, 30, (7, 8, 8), 56), (5, 13, (4, 4, 44), 4)]
        if args.oracle == 'hopper':
            cases += [(17, 448, (37, 15, 27), 4), (3, 11, (4, 4, 8193), 1)]
        for text, audio, shape, heads in cases:
            rows = text + audio + math.prod(shape)
            q, k, v = [torch.randn(1, rows, heads, 128, device='cuda').bfloat16() for _ in range(3)]
            for scale in (0.0, 0.3):
                gate = (torch.randn_like(q) * scale).bfloat16()
                expected = reference(q, k, v, gate, text, audio, shape, upstream,
                                     native.lib if args.oracle == 'hopper' else None).float().cpu()
                context = native.call('vsa_new', text, audio, *shape, heads)
                try:
                    inputs = [native.tensor(x) for x in (q, k, v, gate)]
                    output = native.empty(q.shape, torch.bfloat16)
                    if not native.lib.h3_gpu_vsa_bf16(context, output, *inputs):
                        raise RuntimeError(C.string_at(native.lib.h3_gpu_error(native.gpu)).decode())
                    got = native.read(output, q.shape, torch.bfloat16).float()
                    error = got - expected
                    rel = float(error.norm() / expected.norm())
                    normalized_max = float(error.abs().max() / expected.abs().max())
                    exact = torch.equal(got, expected)
                    passed = bool(torch.isfinite(got).all()) and rel < .012 and normalized_max < .03
                    if args.oracle == 'hopper':
                        passed = passed and exact
                    row = dict(text=text, audio=audio, shape=shape, heads=heads, gate_scale=scale,
                               relative_l2=rel, normalized_max=normalized_max, exact=exact, passed=passed)
                    records.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    native.lib.h3_gpu_vsa_free(context)
                    native.clear()
    finally:
        native.close()
    args.output.write_text(json.dumps(dict(passed=all(r['passed'] for r in records), oracle=args.oracle, cases=records), indent=2))
    if not all(r['passed'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
