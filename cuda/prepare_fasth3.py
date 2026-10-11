#!/usr/bin/env python3
"""Export a FastH3 Preview v1 full transformer for the native C/CUDA runtime.

CPU-only, one block in memory. Reuses the base tokenizer/encoders/VAEs through
symlinks. Does not overwrite either source or an existing output. This is a
layout conversion, not quantization; BF16 boundary weights are exactly widened
to FP32 because that is how the FastVideo inference model executes them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


ARCH = dict(num_attention_heads=56, attention_head_dim=128, hidden_size=5376,
            num_layers=50, num_refiner_layers=2, ffn_dim=14336, in_channels=24,
            audio_in_channels=32, patch_size=[1, 2, 2], text_dim=5120,
            freq_dim=256, time_embed_hidden_dim=5376, time_embed_dim=2688,
            rope_freq_dim=16, rope_theta=10000.0, norm_eps=1e-5,
            qk_norm_eps=1e-5, final_norm_eps=1e-5)
TOP = {
    'proj_in': 'video_patch_proj', 'audio_proj_in': 'audio_patch_proj',
    'context_embedder': 'condition_proj', 'time_embedder.linear_1': 'time_embedder.proj_in',
    'time_embedder.linear_2': 'time_embedder.proj_out', 'norm_out.norm': 'final_layer.norm',
    'norm_out.linear': 'final_layer.adaln_proj.linear', 'proj_out': 'final_layer.video_out',
    'audio_proj_out': 'final_layer.audio_out', 'token_refiner.final_norm': 'token_refiner.final_norm',
}
FP32 = ('video_patch_proj.', 'audio_patch_proj.', 'time_embedder.',
        'final_layer.video_out.', 'final_layer.audio_out.')


def validate_contract(source):
    config = json.loads((source / 'transformer/config.json').read_text())
    for key, value in ARCH.items():
        if config.get(key) != value:
            raise ValueError(f'unsupported architecture: {key}={config.get(key)!r}')
    if config.get('pdd_steps') or config.get('adaln_rank'):
        raise ValueError('PDD and reduced-rank AdaLN checkpoints are not Preview v1')
    c = json.loads((source / 'fastvideo_inference.json').read_text())
    for key, value in dict(schema_version='fasth3-inference-contract-v1', task='t2av',
                           transformer_forwards=4, num_inference_steps=5,
                           dmd_denoising_steps=[999, 749, 500, 250], guidance_scale=1.0).items():
        if c.get(key) != value:
            raise ValueError(f'unsupported inference contract: {key}={c.get(key)!r}')
    backend = c.get('attention_backend')
    if backend not in ('FLASH_ATTN', 'VIDEO_SPARSE_ATTN_H3'):
        raise ValueError(f'unsupported attention backend: {backend}')
    if backend == 'VIDEO_SPARSE_ATTN_H3' and (c.get('vsa_tile_size') != 64 or c.get('vsa_sparsity') != 0.9):
        raise ValueError('expected tile-64, 90% sparse VSA')
    for folder, shift in [('scheduler', 12.0), ('audio_scheduler', 3.0)]:
        if json.loads((source / folder / 'scheduler_config.json').read_text()).get('shift') != shift:
            raise ValueError(f'unsupported {folder} shift')
    return config, c


def convert_group(state):
    import torch
    result, used = {}, set()
    for key, value in state.items():
        if key in used:
            continue
        if key.endswith('.attn.to_q.weight'):
            prefix = key.removesuffix('to_q.weight')
            names = [prefix + f'to_{part}.weight' for part in ('q', 'k', 'v')]
            # Raw h3.c expects [head, q/k/v, channel, input], not [q/k/v, head, channel, input].
            parts = [state[n].reshape(56, 128, 5376) for n in names]
            value = torch.stack(parts, dim=1).reshape(56 * 3 * 128, 5376)
            used.update(names)
            key = prefix + 'qkv_proj.weight'
        elif key.endswith(('.attn.to_k.weight', '.attn.to_v.weight')):
            continue
        if key.startswith('transformer_blocks.'):
            key = key.replace('transformer_blocks.', 'blocks.', 1)
        elif key.startswith('token_refiner.refiner_blocks.'):
            key = key.replace('token_refiner.refiner_blocks.', 'token_refiner.blocks.', 1)
        else:
            parent, suffix = key.rsplit('.', 1)
            if parent not in TOP:
                raise ValueError(f'unrecognized checkpoint tensor: {key}')
            key = TOP[parent] + '.' + suffix
        for old, new in [('.attn.norm_q.', '.attn.q_norm.'), ('.attn.norm_k.', '.attn.k_norm.'),
                         ('.attn.to_out.0.', '.attn.out_proj.'), ('.ff.net.2.', '.mlp.fc2.')]:
            key = key.replace(old, new)
        if '.ff.net.0.proj.' in key:
            # Diffusers stores [value; gate], native H3 stores [gate; value].
            up, gate = value.chunk(2, dim=0)
            value = torch.cat((gate, up), dim=0)
            key = key.replace('.ff.net.0.proj.', '.mlp.fc1.')
        value = value.to(torch.float32 if key.startswith(FP32) else torch.bfloat16).contiguous()
        if key in result:
            raise ValueError(f'duplicate converted tensor: {key}')
        result[key] = value
    return result


def write_rope_buffer(target, config):
    """Diffusers omits this derived buffer; the native weight store requires it."""
    import torch
    from safetensors.torch import save_file
    dim = config['rope_freq_dim']
    inverse = 1.0 / (config['rope_theta'] ** (torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim)))
    filename = 'model-rope.safetensors'
    save_file({'rope.inv_freq': inverse}, target / filename)
    digest = hashlib.sha256((target / filename).read_bytes()).hexdigest()
    return filename, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='full FastH3 snapshot root (transformer and JSONs suffice)')
    parser.add_argument('--base', type=Path, required=True, help='original MiniMax-H3 root, containing FL2VA')
    parser.add_argument('--output', type=Path, required=True, help='new native model root')
    args = parser.parse_args()
    config, contract = validate_contract(args.source)
    index_path = args.source / 'transformer/diffusion_pytorch_model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map']
    gates = [k for k in weight_map if k.endswith('.attn.to_gate_compress.weight')]
    vsa = contract['attention_backend'] == 'VIDEO_SPARSE_ATTN_H3'
    if len(gates) != (50 if vsa else 0) or len(weight_map) != (688 if vsa else 638):
        raise ValueError('unexpected parameter inventory; refusing to drop checkpoint tensors')
    base = (args.base / 'FL2VA').resolve(strict=True)
    for name in ('tokenizer', 'text_encoder', 'video_vae', 'audio_vae'):
        if not (base / name).is_dir():
            raise ValueError(f'missing base component: {name}')
    if args.output.exists():
        raise ValueError('output already exists; choose a new directory')
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    target = args.output / 'FL2VA/transformer'
    target.mkdir(parents=True)
    groups = {}
    for key in sorted(weight_map):
        m = re.match(r'(transformer_blocks\.\d+|token_refiner\.refiner_blocks\.\d+)\.', key)
        groups.setdefault(m[1] if m else 'heads', []).append(key)
    output_map, hashes = {}, {}
    for i, (group, keys) in enumerate(groups.items()):
        state = {}
        for filename in sorted({weight_map[k] for k in keys}):
            with safe_open(args.source / 'transformer' / filename, framework='pt', device='cpu') as f:
                for key in keys:
                    if weight_map[key] == filename:
                        state[key] = f.get_tensor(key)
                        if state[key].dtype not in (torch.bfloat16, torch.float32) or not torch.isfinite(state[key]).all():
                            raise ValueError(f'invalid weight: {key}')
        converted = convert_group(state)
        filename = f'model-{i + 1:05d}-of-{len(groups):05d}.safetensors'
        save_file(converted, target / filename)
        output_map.update({k: filename for k in converted})
        with (target / filename).open('rb') as f:
            hashes[filename] = hashlib.file_digest(f, 'sha256').hexdigest()
        print(f'{group}: {len(keys)} tensors -> {filename}', flush=True)
        del state, converted
    filename, digest = write_rope_buffer(target, config)
    output_map['rope.inv_freq'] = filename
    hashes[filename] = digest
    (target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (target / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': output_map}, indent=2) + '\n')
    manifest = dict(format='h3.c-fasth3-v1', task='t2av', attention='vsa' if vsa else 'dense',
                    guidance_scale=1.0, dmd_denoising_steps=contract['dmd_denoising_steps'],
                    video_shift=12.0, audio_shift=3.0, tile_size=64, sparsity=0.9,
                    source_contract=contract, shards_sha256=hashes)
    (target / 'h3_fasth3.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name in ('tokenizer', 'text_encoder', 'video_vae', 'audio_vae'):
        (target.parent / name).symlink_to(base / name, target_is_directory=True)
    print(f'Native model: {args.output}', flush=True)


if __name__ == '__main__':
    main()
