"""Replay output normalization and projections on a captured final hidden state."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
from native import Native
from real_dit import conversion_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--diffusers', required=True)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    conversion = conversion_module(args.diffusers)
    config = conversion.MINIMAX_H3_TRANSFORMER_CONFIG
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel(**config)
    state = {}
    for path in sorted((args.weights/'transformer').glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as source:
            for key in source.keys():
                if key.startswith(('time_embedder.', 'final_layer.')):
                    for target, value in conversion.convert_transformer_key(key, source.get_tensor(key), config):
                        state[target] = value.cuda()
    model.load_state_dict(state, strict=False, assign=True)
    model.eval()
    inputs = load_file(str(args.trace/'inputs.safetensors'))
    with safe_open(args.trace/'trace-reference.safetensors', framework='pt', device='cpu') as source:
        hidden = source.get_tensor('block-49').cuda()
    _, tags, *_ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(inputs['text'].shape[1], dtype=torch.long), *inputs['video'].shape[2:],
        inputs['audio'].shape[-1], (1,2,2), 2, 2, 0)
    indices = (tags == 2).long().cuda()
    rows, width = hidden.shape[1:]
    native = Native('build-cuda/libh3_cuda.so')
    records = []

    def compare(label, got, expected):
        got, expected = got.cpu().float(), expected.cpu().float()
        delta = got - expected
        row = dict(stage=label, relative_l2=float(delta.norm()/expected.norm().clamp_min(1e-12)),
                   normalized_max=float(delta.abs().max()/expected.abs().max().clamp_min(1e-12)),
                   unequal_fraction=float((got != expected).float().mean()))
        records.append(row)
        print(json.dumps(row), flush=True)

    def operation(label, name, expected, values, scalars):
        output = native.empty(expected.shape, expected.dtype)
        pointers = [native.tensor(v) if v is not None else None for v in values]
        native.call(name, output, *pointers, *scalars)
        got = native.read(output, expected.shape, expected.dtype)
        compare(label, got, expected)
        native.clear()
        return got

    with torch.inference_mode():
        temb = model.time_embedder(model.time_proj(torch.zeros(2,device='cuda')))
        modulation = model.norm_out.linear(F.silu(temb).bfloat16())
        operation('modulation', 'linear_bf16', modulation,
                  [F.silu(temb).bfloat16(), model.norm_out.linear.weight, model.norm_out.linear.bias],
                  [2, 2688, width*2])
        normalized = model.norm_out(hidden, temb, indices)
        operation('normalization', 'adaln_bf16', normalized,
                  [hidden, model.norm_out.norm.weight, modulation, indices.int()],
                  [rows, width, 2, 0, 1, 1e-5])
        x = normalized.float()
        for name, head in (('video', model.proj_out), ('audio', model.audio_proj_out)):
            expected = head(x)
            operation(name+'_head', 'linear_f32', expected,
                      [x, head.weight, head.bias], [rows, width, head.out_features])
            compare(name+'_separate_bias', F.linear(x, head.weight)+head.bias, expected)
    native.close()
    args.output.write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
