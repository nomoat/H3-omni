#!/usr/bin/env python3
"""Export TaoMate-H3 runtime LoRA buffers. Base weights remain unchanged.

The released inference casts FP32 adapters to BF16 once. Keep the two LoRA
products separate from the base GEMM to preserve that rounding contract.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    from safetensors import safe_open
    from safetensors.torch import save_file
    base, adapter, output = a.base.resolve(), a.adapter.resolve(), a.output.resolve()
    if output.exists():
        raise ValueError('output must not already exist')
    source = base / 'FL2VA/transformer'
    if (source / 'h3_fasth3.json').exists() or (source / 'h3_live.json').exists():
        raise ValueError('TaoMate requires the original H3 base checkpoint')
    config = json.loads((adapter / 'config.json').read_text())
    if config.get('rank') != 128 or config.get('alpha') != 128:
        raise ValueError('expected released rank-128, alpha-128 adapter')
    shapes = {'attn.qkv_proj': (21504, 5376), 'attn.out_proj': (5376, 7168),
              'mlp.fc1': (28672, 5376), 'mlp.fc2': (5376, 14336)}
    prefixes = [f'blocks.{i}' for i in range(50)] + [f'token_refiner.blocks.{i}' for i in range(2)]
    expected = {}
    for prefix in prefixes:
        for target, (n, k) in shapes.items():
            expected[f'{prefix}.{target}.lora_a'] = (128, k)
            expected[f'{prefix}.{target}.lora_b'] = (n, 128)
    path = adapter / 'adapter_model.safetensors'
    buffers = {}
    with safe_open(path, framework='pt', device='cpu') as f:
        if set(f.keys()) != set(expected):
            raise ValueError('adapter must contain the 416 canonical H3 tensors')
        for key, shape in expected.items():
            t = f.get_tensor(key)
            if tuple(t.shape) != shape or t.dtype != torch.float32 or not t.isfinite().all():
                raise ValueError(f'invalid adapter tensor {key}')
            # TaoMate trains against [Q heads, K heads, V heads], whereas
            # the native base checkpoint keeps [head, Q/K/V, dimension].
            if key.endswith('.attn.qkv_proj.lora_b'):
                t = t.reshape(3, 56, 128, 128).permute(1, 0, 2, 3).reshape(21504, 128)
            buffers[key] = t.to(torch.bfloat16).contiguous()
    target = output / 'FL2VA/transformer'
    target.mkdir(parents=True)
    for child in base.iterdir():
        if child.name != 'FL2VA':
            (output / child.name).symlink_to(child, target_is_directory=child.is_dir())
    for child in (base / 'FL2VA').iterdir():
        if child.name != 'transformer':
            (output / 'FL2VA' / child.name).symlink_to(child, target_is_directory=child.is_dir())
    for child in source.iterdir():
        if child.name != 'model.safetensors.index.json':
            (target / child.name).symlink_to(child)
    save_file(buffers, target / 'taomate-lora.safetensors')
    index_path = source / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    index['weight_map'].update({key: 'taomate-lora.safetensors' for key in buffers})
    index.setdefault('metadata', {})['total_size'] = sum(
        t.numel() * t.element_size() for t in buffers.values()) + int(index.get('metadata', {}).get('total_size', 0))
    (target / index_path.name).write_text(json.dumps(index, indent=2) + '\n')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            digest.update(chunk)
    manifest = dict(format='h3.c-live-v1', model='TaoMate-H3', rank=128, alpha=128,
                    state_indices=[0, 16, 33, 49], video_shift=12, audio_shift=3,
                    qkv_layout='head-qkv-dim', base=str(base), adapter_sha256=digest.hexdigest(),
                    upstream_commit='6b2f9984acf681e2f387188f5d1f3b26e504315b')
    (target / 'h3_live.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
