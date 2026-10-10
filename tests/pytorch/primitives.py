"""Differential CUDA tests. The reference computations run in PyTorch on H200."""
import argparse
import json
import math
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from native import Native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build-cuda/libh3_cuda.so")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    native = Native(args.library)
    limits = json.loads(Path(__file__).with_name("acceptance.json").read_text())["primitives"]
    records = []

    def check(name, shape, dtype, function, inputs, scalars, reference):
        record = {"name": name, "shape": list(shape), "dtype": str(dtype)}
        try:
            gpu_inputs = [x.cuda() for x in inputs]
            with torch.no_grad():
                expected = reference(*gpu_inputs).detach().cpu().float()
            actual_ptr = native.empty(shape, dtype)
            pointers = [native.tensor(x) for x in inputs]
            native.call(function, actual_ptr, *pointers, *scalars)
            actual = native.read(actual_ptr, shape, dtype).float()
            difference = actual - expected
            rel = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected).clamp_min(1e-12))
            threshold = limits[str(dtype).removeprefix("torch.")]
            close = torch.allclose(actual, expected, atol=threshold["atol"], rtol=threshold["rtol"])
            record.update(max_abs=float(difference.abs().max()), relative_l2=rel,
                          passed=bool(close and rel <= threshold["relative_l2"] and torch.isfinite(actual).all()))
        except Exception:
            record.update(passed=False, error=traceback.format_exc())
        finally:
            native.clear()
        print(json.dumps(record), flush=True)
        records.append(record)

    def time_features(times):
        frequency = torch.exp(-math.log(10000) * torch.arange(128, device=times.device).float() / 128)
        angles = times[:,None] * frequency[None,:]
        return torch.cat((angles.cos(), angles.sin()), dim=1)
    check("timestep_features", (4,256), torch.float32, "time_features_f32",
          [torch.tensor([0., .123, .5, .99])], [4,256], time_features)
    for rows in (2,8,11):
        x = torch.randn(rows,5120).bfloat16()
        w = (torch.randn(5120)*.1+1).bfloat16()
        check("qwen_hidden_norm", x.shape, torch.bfloat16, "qwen_rms_norm_bf16", [x,w], [rows,5120,1e-6],
              lambda x,w: (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()*w)

    for dtype in (torch.float32, torch.bfloat16):
        suffix = "f32" if dtype == torch.float32 else "bf16"
        for rows, width in ((1, 7), (17, 128), (3, 5376)):
            x = torch.randn(rows, width).to(dtype)
            w = (torch.randn(width) * 0.1 + 1).to(dtype)
            check("silu", x.shape, dtype, "silu_" + suffix, [x], [x.numel()], F.silu)
            check("rms_norm", x.shape, dtype, "rms_norm_" + suffix, [x, w], [rows, width, 1e-6],
                  lambda x, w: F.rms_norm(x, (width,), w, 1e-6))
            if dtype == torch.bfloat16:
                check("qwen_rms_norm", x.shape, dtype, "qwen_rms_norm_bf16", [x, w], [rows, width, 1e-6],
                      lambda x, w: (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w)
            bias = torch.randn(width).to(dtype) * 0.1
            check("layer_norm", x.shape, dtype, "layer_norm_" + suffix, [x, w, bias], [rows, width, 1e-6],
                  lambda x, w, b: F.layer_norm(x, (width,), w, b, 1e-6))
            fused = torch.randn(rows, width * 2).to(dtype)
            check("swiglu", x.shape, dtype, "swiglu_" + suffix, [fused], [rows, width],
                  lambda a: F.silu(a[..., :width]) * a[..., width:])
            modulation = (torch.randn(3, 6, width) * 0.1).to(dtype)
            row_map = torch.arange(rows, dtype=torch.int32) % 3
            check("adaln", x.shape, dtype, "adaln_" + suffix,
                  [x, w, modulation, row_map], [rows, width, 6, 0, 1, 1e-6],
                  lambda x, w, m, idx: F.rms_norm(x, (width,), w, 1e-6) * (1 + m[idx.long(), 1]) + m[idx.long(), 0])
            branch = torch.randn_like(x)
            check("gate", x.shape, dtype, "gate_" + suffix,
                  [x, branch, modulation, row_map], [rows, width, 6, 2],
                  lambda x, b, m, idx: x + b * m[idx.long(), 2])
        for rows, k, n in ((1, 7, 9), (17, 128, 256), (24, 5376, 96)):
            x = torch.randn(rows, k).to(dtype)
            w = (torch.randn(n, k) / k ** 0.5).to(dtype)
            b = (torch.randn(n) * 0.1).to(dtype)
            check("linear_bias", (rows, n), dtype, "linear_" + suffix, [x, w, b], [rows, k, n], F.linear)
        for sequence, heads, dim in ((7, 2, 8), (263, 3, 32), (24, 56, 128)):
            q, k, v = [torch.randn(sequence, heads, dim).to(dtype) for _ in range(3)]
            def attention(q, k, v):
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    return F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1)
            check("sdpa", q.shape, dtype, "sdpa_" + suffix, [q, k, v], [sequence, heads, dim, dim ** -0.5], attention)

    # Four-dimensional inputs select PyTorch's real FlashAttention path.
    # Odd lengths exercise masked tiles; 128 exercises the unmasked path.
    for sequence in (7, 128, 263):
        for causal in (False, True):
            heads, kv_heads = 8, 2 if causal else 8
            q = torch.randn(sequence, heads, 128).bfloat16()
            k, v = [torch.randn(sequence, kv_heads, 128).bfloat16() for _ in range(2)]
            def flash_reference(q, k, v):
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
                        k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
                        is_causal=causal, enable_gqa=causal).squeeze(0).transpose(0, 1)
            function = "gqa_causal_bf16" if causal else "sdpa_bf16"
            scalars = [sequence, heads, kv_heads, 128, 128 ** -.5] if causal else [sequence, heads, 128, 128 ** -.5]
            check("flash_gqa_causal" if causal else "flash_sdpa", q.shape, torch.bfloat16,
                  function, [q, k, v], scalars, flash_reference)

    for sequence in (8, 263):
        q = torch.randn(sequence, 8, 128).bfloat16()
        k, v = [torch.randn(sequence, 2, 128).bfloat16() for _ in range(2)]
        def gqa(q, k, v):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                    v.transpose(0, 1), is_causal=True, enable_gqa=True).transpose(0, 1)
        check("gqa_causal", q.shape, torch.bfloat16, "gqa_causal_bf16", [q, k, v],
              [sequence, 8, 2, 128, 128 ** -0.5], gqa)
    q, k, v = [torch.randn(2, 19, 8, 64) for _ in range(3)]
    def batched_causal(q, k, v):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), is_causal=True).transpose(1, 2)
    check("sdpa_causal_batch", q.shape, torch.float32, "sdpa_causal_f32", [q, k, v],
          [2, 19, 8, 64, 64 ** -0.5], batched_causal)

    for stride, dilation in ((1, 1), (2, 1), (1, 3)):
        batch, length, ci, co, kernel, pad = 2, 37, 5, 7, 3, 1
        x = torch.randn(batch, length, ci)
        w = torch.randn(co, ci, kernel) * 0.1
        b = torch.randn(co) * 0.1
        outlen = (length + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
        check("conv1d", (batch, outlen, co), torch.float32, "conv1d_stride_f32", [x, w, b],
              [batch, length, ci, co, kernel, stride, pad, dilation],
              lambda x, w, b: F.conv1d(x.transpose(1, 2), w, b, stride, pad, dilation).transpose(1, 2))
    x, w, b = torch.randn(2, 17, 5), torch.randn(5, 7, 4) * .1, torch.randn(7) * .1
    check("conv_transpose1d", (2, 34, 7), torch.float32, "conv_transpose1d_f32", [x, w, b],
          [2, 17, 5, 7, 4, 2, 1], lambda x, w, b: F.conv_transpose1d(x.transpose(1, 2), w, b, 2, 1).transpose(1, 2))
    x, w, b = torch.randn(1, 5, 7, 8, 3), torch.randn(4, 3, 2, 3, 3) * .1, torch.randn(4) * .1
    check("conv3d", (1, 4, 3, 3, 4), torch.float32, "conv3d_f32", [x, w, b],
          [1, 5, 7, 8, 3, 4, 2, 3, 3, 1, 2, 2], lambda x, w, b: F.conv3d(x.permute(0, 4, 1, 2, 3), w, b, (1, 2, 2)).permute(0, 2, 3, 4, 1))
    native.close()
    report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "passed": all(x["passed"] for x in records), "cases": records}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
