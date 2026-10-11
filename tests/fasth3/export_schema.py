"""Check every converted name, shape and dtype against the native base schema.

Reads safetensors headers only. Meta tensors exercise conversion without
allocating the full checkpoint or reading weight data.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import struct

import torch


def headers(folder):
    result = {}
    for path in sorted(folder.glob('*.safetensors')):
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        for key, value in header.items():
            if key == '__metadata__':
                continue
            assert key not in result, key
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / 'cuda/prepare_fasth3.py'
    spec = importlib.util.spec_from_file_location('exporter', source)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.validate_contract(args.source)
    types = {'F32': torch.float32, 'BF16': torch.bfloat16}
    state = {key: torch.empty(value['shape'], dtype=types[value['dtype']], device='meta')
             for key, value in headers(args.source / 'transformer').items()}
    output = exporter.convert_group(state)
    output['rope.inv_freq'] = torch.empty(16, dtype=torch.float32, device='meta')
    expected = headers(args.base / 'FL2VA/transformer')
    gates = {f'blocks.{i}.attn.to_gate_compress.weight' for i in range(50)}
    assert set(output) == set(expected) | gates, (set(output) - set(expected) - gates, set(expected) - set(output))
    for key, value in output.items():
        shape = [7168, 5376] if key in gates else expected[key]['shape']
        dtype = torch.bfloat16 if key in gates else types[expected[key]['dtype']]
        assert list(value.shape) == shape and value.dtype == dtype, (key, value.shape, value.dtype, shape, dtype)
    report = dict(passed=True, source_tensors=len(state), native_tensors=len(output), learned_gates=len(gates),
                  native_parameters=sum(v.numel() for v in output.values()))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
