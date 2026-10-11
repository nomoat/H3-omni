"""Four-step native FastH3 vs Diffusers + PyTorch VSA orchestration.

Reuses the original exact packing/sampler and trajectory checks. Loads the
unconverted checkpoint for the reference. Only attention changes: the official
FastVideo tile/mask helpers feed either the upstream Hopper kernel or a
diagnostic FP32 math SDPA. The shared Hopper kernel is tested separately with
the independent math oracle in vsa.py. This does not claim validation of the
entire FastVideo application.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pytorch'))
import real_dit
from vsa import reference as vsa_reference, upstream_helpers
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb


class DistilledScheduler(real_dit.MiniMaxH3Scheduler):
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
        if num_inference_steps != 5 or sigmas is not None:
            raise ValueError('FastH3 Preview v1 requires four evaluations (five sigma points)')
        base = torch.tensor([step / 1000.0 for step in [999, 749, 500, 250]] + [0.0], dtype=torch.float32)
        sigmas = float(self.shift) * base / (1 + (float(self.shift) - 1) * base)
        return super().set_timesteps(sigmas=sigmas, device=device)


class VSAProcessor:
    def __init__(self, geometry, upstream, hopper=None):
        self.geometry, self.upstream, self.hopper = geometry, upstream, hopper
        self.text, self.audio = None, None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        assert attention_mask is None and self.text is not None
        q, k, v = [layer(hidden_states).unflatten(-1, (attn.heads, -1))
                   for layer in (attn.to_q, attn.to_k, attn.to_v)]
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        gate = attn.to_gate_compress(hidden_states).unflatten(-1, (attn.heads, -1))
        out = vsa_reference(q, k, v, gate, self.text, self.audio, self.geometry, self.upstream, self.hopper)
        if os.environ.get('H3_CUDA_TRACE_DIR') and not getattr(self, 'traced', False):
            save_file({name: value.cpu().contiguous() for name, value in
                       [('vsa-input', hidden_states), ('vsa-q', q), ('vsa-k', k), ('vsa-v', v),
                        ('vsa-gate', gate), ('vsa-heads', out)]},
                      str(Path(os.environ['H3_CUDA_TRACE_DIR']).parent / 'vsa-reference.safetensors'))
            self.traced = True
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fastvideo', type=Path, required=True)
    parser.add_argument('--oracle', choices=['math', 'hopper'], default='hopper',
                        help='math: independent FP32 attention; hopper: shared upstream kernel, independent integration')
    args, rest = parser.parse_known_args()
    geometry_parser = argparse.ArgumentParser(add_help=False)
    geometry_parser.add_argument('--width', type=int, default=256)
    geometry_parser.add_argument('--height', type=int, default=256)
    geometry_parser.add_argument('--frames', type=int, default=22)
    geometry, _ = geometry_parser.parse_known_args(rest)
    shape = (real_dit.video_latent_num_frames(geometry.frames, 17, 5),
             geometry.height // 32, geometry.width // 32)
    upstream = upstream_helpers(args.fastvideo)
    source = args.source / 'transformer'
    config = json.loads((source / 'config.json').read_text())
    keep_fp32 = ('proj_in.', 'audio_proj_in.', 'time_embedder.', 'proj_out.', 'audio_proj_out.')

    def load_reference(_weights, conversion):
        with torch.device('meta'):
            model = real_dit.MiniMaxH3Transformer3DModel(**conversion.MINIMAX_H3_TRANSFORMER_CONFIG)
            for block in model.transformer_blocks:
                block.attn.to_gate_compress = torch.nn.Linear(5376, 7168, bias=False)
        state = {}
        index = json.loads((source / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        for filename in sorted(set(index.values())):
            print('Reference loading', filename, flush=True)
            with safe_open(source / filename, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key not in index:
                        raise ValueError('unindexed source tensor: ' + key)
                    dtype = torch.float32 if key.startswith(keep_fp32) else torch.bfloat16
                    state[key] = f.get_tensor(key).to(device='cuda', dtype=dtype)
        model.load_state_dict(state, strict=True, assign=True)
        model.rope = real_dit.MiniMaxH3RotaryPosEmbed(config['rope_freq_dim'], config['rope_theta']).cuda()
        hopper = C.CDLL(str(Path('build-cuda/libh3_cuda.so').resolve())) if args.oracle == 'hopper' else None
        processor = VSAProcessor(shape, upstream, hopper)
        for block in model.transformer_blocks:
            block.attn.set_processor(processor)

        def set_layout(_model, _args, kwargs):
            processor.text = kwargs['text_indices'].numel()
            processor.audio = kwargs['audio_indices'].numel()
        model.register_forward_pre_hook(set_layout, with_kwargs=True)
        return model.eval()

    real_dit.load_reference = load_reference
    real_dit.MiniMaxH3Scheduler = DistilledScheduler
    cache = {}
    try:
        report = real_dit.main(rest + ['--steps', '4'], cache=cache)
        report['reference'] = 'Pinned Diffusers model + FastVideo tile/mask helpers + PyTorch pooling/gating'
        report['sparse_kernel_oracle'] = args.oracle
        report['source_checkpoint'] = str(args.source)
        output = Path(report['arguments']['output']) / 'results.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        if 'native' in cache:
            cache['native'].close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
