"""Isolate numerical differences using real block-zero weights and activations."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb, MiniMaxH3RotaryPosEmbed
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
from native import Native
from real_dit import conversion_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
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
    state, raw = {}, {}
    for path in sorted((Path(args.weights)/'transformer').glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as source:
            for key in source.keys():
                if not key.startswith(('blocks.0.', 'time_embedder.')):
                    continue
                value = source.get_tensor(key).cuda()
                raw[key] = value
                if key.endswith('.attn.qkv_proj.weight'):
                    value = conversion.reorder_interleaved_qkv(value, config['num_attention_heads'], config['attention_head_dim'])
                for target, value in conversion.convert_transformer_key(key, value, config):
                    state[target] = value
    model.load_state_dict(state, strict=False, assign=True)
    model.eval()
    block = model.transformer_blocks[0]
    inputs = load_file(str(args.trace/'inputs.safetensors'))
    with safe_open(args.trace/'trace-reference.safetensors', framework='pt', device='cpu') as source:
        traces = {key: source.get_tensor(key) for key in ('packed_input', 'block-00')}
    x = traces['packed_input'].cuda()
    video = inputs['video']
    positions, tags, *_ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(inputs['text'].shape[1], dtype=torch.long), *video.shape[2:],
        inputs['audio'].shape[-1], (1,2,2), 2, 2, 0)
    indices = (tags + torch.where(tags == 2, 3, 0)).cuda()
    rope = MiniMaxH3RotaryPosEmbed(config['rope_freq_dim'], config['rope_theta']).cuda()(positions.cuda())
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

    native_input = torch.frombuffer(bytearray((args.trace / 'trace-native' / 'packed_input.bf16').read_bytes()),
                                   dtype=torch.bfloat16).reshape(x.shape)
    for tag, label in ((0, 'video'), (1, 'text'), (2, 'audio')):
        compare('packed_input_' + label, native_input[:,tags == tag], x.cpu()[:,tags == tag])
    for name, expected in zip(('rope-cos', 'rope-sin'), rope):
        path = args.trace / 'trace-native' / (name + '.bf16')
        if path.exists():
            actual = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(-1,48)
            compare(name, actual, expected[:,:48].bfloat16())

    def operation(label, name, expected, values, scalars):
        output = native.empty(expected.shape, expected.dtype)
        pointers = [native.tensor(v) if v is not None else None for v in values]
        native.call(name, output, *pointers, *scalars)
        got = native.read(output, expected.shape, expected.dtype)
        compare(label, got, expected)
        native.clear()
        return got

    with torch.inference_mode():
        features = model.time_proj(torch.zeros(2, device='cuda'))
        first = model.time_embedder.linear_1(features)
        operation('time_linear1', 'linear_f32', first,
                  [features, model.time_embedder.linear_1.weight, model.time_embedder.linear_1.bias],
                  [2,256,5376])
        operation('time_silu', 'silu_f32', F.silu(first), [first], [first.numel()])
        temb = model.time_embedder(features)
        operation('time_linear2', 'linear_f32', temb,
                  [F.silu(first), model.time_embedder.linear_2.weight, model.time_embedder.linear_2.bias],
                  [2,5376,2688])
        for batch in (2,39):
            first_native = operation(f'time_chain_linear1_{batch}', 'linear_f32', first[:1].expand(batch,-1),
                [features[:1].expand(batch,-1),model.time_embedder.linear_1.weight,model.time_embedder.linear_1.bias],
                [batch,256,5376])
            act_native = operation(f'time_chain_silu_{batch}', 'silu_f32', F.silu(first[:1]).expand(batch,-1),
                                   [first_native], [batch*5376])
            second_native = operation(f'time_chain_linear2_{batch}', 'linear_f32', temb[:1].expand(batch,-1),
                [act_native,model.time_embedder.linear_2.weight,model.time_embedder.linear_2.bias],
                [batch,5376,2688])
            final_native = operation(f'time_chain_activation_{batch}', 'silu_f32', F.silu(temb[:1]).expand(batch,-1),
                                     [second_native], [batch*2688])
            compare(f'time_chain_bf16_{batch}', final_native.bfloat16(), F.silu(temb[:1]).bfloat16().expand(batch,-1))
        modulation = block.adaln_proj(temb)
        combined = torch.cat(modulation, -1).reshape(2, -1)
        def read_trace(name, width):
            path = args.trace / 'trace-native' / (name + '.bf16')
            if path.exists():
                return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(-1,width)
        actual_time = read_trace('time-activated', 2688)
        if actual_time is not None:
            compare('actual_time_embedding', actual_time[:1], F.silu(temb).bfloat16()[:1])
        actual_modulation = read_trace('modulation-00', 6*5376*3)
        if actual_modulation is not None:
            compare('actual_modulation_table', actual_modulation[:1], combined[:1])
        operation('modulation', 'linear_bf16', combined,
                  [F.silu(temb).bfloat16(), block.adaln_proj.linear.weight, block.adaln_proj.linear.bias],
                  [2, 2688, 6*5376*3])
        rows, width = x.shape[1:]
        normalized = block.norm1(x)
        operation('norm1', 'rms_norm_bf16', normalized, [x, block.norm1.weight], [rows, width, 1e-5])
        modulated = normalized * (1 + modulation[1][indices]) + modulation[0][indices]
        operation('adaln1', 'adaln_bf16', modulated,
                  [x, block.norm1.weight, combined, indices.int()], [rows, width, 6, 0, 1, 1e-5])
        q, k, v = [layer(modulated).unflatten(-1, (56,128)) for layer in (block.attn.to_q, block.attn.to_k, block.attn.to_v)]
        for label, layer, expected in zip(('query_projection', 'key_projection', 'value_projection'),
                                         (block.attn.to_q, block.attn.to_k, block.attn.to_v), (q,k,v)):
            operation(label, 'linear_bf16', expected, [modulated, layer.weight, None], [rows,width,7168])
        grouped = torch.stack((q,k,v), -2).reshape(1,rows,-1)
        operation('grouped_qkv_projection', 'linear_bf16', grouped,
                  [modulated, raw['blocks.0.attn.qkv_proj.weight'], None], [rows, width, 56*128*3])
        qr, kr = [_apply_rotary_emb(norm(tensor), *rope) for tensor,norm in ((q,block.attn.norm_q),(k,block.attn.norm_k))]
        outputs = [native.empty(t.shape,t.dtype) for t in (qr,kr,v)]
        values = [grouped, block.attn.norm_q.weight, block.attn.norm_k.weight,
                  rope[0][:,:48].bfloat16(), rope[1][:,:48].bfloat16()]
        native.call('grouped_qkv_rope_bf16', *outputs, *[native.tensor(t) for t in values], rows, 56, 128, 48, 1e-5)
        for label,pointer,expected in zip(('query_norm_rope','key_norm_rope','value_split'), outputs, (qr,kr,v)):
            compare(label,native.read(pointer,expected.shape,expected.dtype),expected)
        native.clear()
        attention = F.scaled_dot_product_attention(qr.transpose(1,2),kr.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        got_attention = operation('attention_default', 'sdpa_bf16', attention, [qr,kr,v], [rows,56,128,128**-.5])
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            math_attention = F.scaled_dot_product_attention(qr.transpose(1,2),kr.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        compare('attention_math',got_attention,math_attention)
        compare('pytorch_default_vs_math_attention',attention,math_attention)
        # Diagnostics only: locate the probability rounding used by fused SDPA.
        qf, kf, vf = [t.transpose(1,2).float() for t in (qr,kr,v)]
        scores = (qf @ kf.transpose(-1,-2)) * 128**-.5
        probabilities = scores.softmax(-1)
        compare('attention_rounded_normalized_probability',
                (probabilities.bfloat16().float() @ vf).bfloat16().transpose(1,2), attention)
        for tile in (64,128,256):
            maximum = torch.full((*scores.shape[:-1],1), -torch.inf, device='cuda')
            total = torch.zeros_like(maximum)
            accumulated = torch.zeros_like(qf)
            for start in range(0,rows,tile):
                score = scores[...,start:start+tile]
                new_maximum = torch.maximum(maximum, score.amax(-1,keepdim=True))
                correction = (maximum-new_maximum).exp()
                probability = (score-new_maximum).exp()
                accumulated = accumulated*correction + probability.bfloat16().float() @ vf[...,start:start+tile,:]
                total = total*correction + probability.sum(-1,keepdim=True)
                maximum = new_maximum
            compare(f'attention_rounded_online_tile_{tile}',
                    (accumulated/total).bfloat16().transpose(1,2),attention)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            F.scaled_dot_product_attention(qr.transpose(1,2),kr.transpose(1,2),v.transpose(1,2))
        print('PyTorch attention dispatch:', [e.key for e in profile.key_averages() if 'scaled_dot' in e.key], flush=True)
        attention = attention.flatten(2)
        branch = block.attn.to_out[0](attention)
        operation('attention_projection','linear_bf16',branch,[attention,block.attn.to_out[0].weight,None],[rows,7168,width])
        hidden = x + modulation[2][indices]*branch
        operation('attention_gate','gate_bf16',hidden,[x,branch,combined,indices.int()],[rows,width,6,2])
        mlp_input = block.norm2(hidden)*(1+modulation[4][indices])+modulation[3][indices]
        operation('adaln2','adaln_bf16',mlp_input,[hidden,block.norm2.weight,combined,indices.int()],[rows,width,6,3,4,1e-5])
        mlp_output=block.ff(mlp_input)
        fc1 = F.linear(mlp_input, raw['blocks.0.mlp.fc1.weight'])
        operation('mlp_fc1', 'linear_bf16', fc1,
                  [mlp_input,raw['blocks.0.mlp.fc1.weight'],None],[rows,width,28672])
        activation = F.silu(fc1[...,:14336]) * fc1[...,14336:]
        operation('mlp_swiglu', 'swiglu_bf16', activation, [fc1], [rows,14336])
        operation('mlp_fc2', 'linear_bf16', mlp_output,
                  [activation,raw['blocks.0.mlp.fc2.weight'],None],[rows,14336,width])
        operation('mlp','mlp_bf16',mlp_output,[mlp_input,raw['blocks.0.mlp.fc1.weight'],raw['blocks.0.mlp.fc2.weight']],[rows,width,14336,width])
        compare('reference_replay',block(x,temb,indices,rope),traces['block-00'])
    native.close()
    args.output.write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
