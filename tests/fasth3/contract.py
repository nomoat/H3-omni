"""Exact DMD schedule/Euler checks and fail-closed checkpoint manifest tests."""
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pytorch'))
from host import Schedule


class Config(C.Structure):
    _fields_ = [('kind', C.c_int), ('steps', C.c_int), ('rungs', C.c_int * 4),
                ('video_shift', C.c_float), ('audio_shift', C.c_float)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', default='build-cuda/libh3_model.so')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib = C.CDLL(str(Path(args.library).resolve()))
    lib.h3_fasth3_config_read.argtypes = [C.c_char_p, C.POINTER(Config), C.c_void_p, C.c_size_t]
    lib.h3_fasth3_schedule_build.argtypes = [C.POINTER(Config), C.POINTER(Schedule)]
    lib.h3_euler_velocity_step.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_float, C.c_float]
    manifest = dict(format='h3.c-fasth3-v1', task='t2av', attention='vsa', guidance_scale=1.0,
                    dmd_denoising_steps=[999, 749, 500, 250], video_shift=12.0, audio_shift=3.0,
                    tile_size=64, sparsity=.9)
    records = []
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as folder:
        config, error = Config(), C.create_string_buffer(1024)
        read = lambda: lib.h3_fasth3_config_read(folder.encode(), C.byref(config), error, len(error))
        assert read() and config.kind == 0
        records.append(dict(case='missing manifest keeps original H3', passed=True))
        path = Path(folder) / 'h3_fasth3.json'
        for attention in ('vsa', 'dense'):
            path.write_text(json.dumps({**manifest, 'attention': attention}))
            assert read() and config.kind == (2 if attention == 'vsa' else 1)
            schedule = Schedule()
            assert lib.h3_fasth3_schedule_build(C.byref(config), C.byref(schedule))
            assert schedule.steps == 4
            for modality, shift in [('video', 12.0), ('audio', 3.0)]:
                base = torch.tensor([step / 1000.0 for step in [999, 749, 500, 250]] + [0.0])
                sigmas = shift * base / (1 + (shift - 1) * base)
                actual = torch.tensor(list(getattr(schedule, modality))[:5])
                assert torch.equal(actual, sigmas)
                reference = MiniMaxH3Scheduler(shift=shift)
                reference.set_timesteps(sigmas=sigmas)
                sample = torch.randn(513)
                expected = sample.clone()
                for step in range(4):
                    velocity = torch.randn_like(sample)
                    assert lib.h3_euler_velocity_step(sample.data_ptr(), velocity.data_ptr(), sample.numel(),
                                                     actual[step].item(), actual[step + 1].item())
                    expected = reference.step(velocity, reference.timesteps[step], expected).prev_sample
                    assert torch.equal(sample, expected)
                records.append(dict(case=f'{attention}/{modality}: exact four-step sigmas and Euler', passed=True,
                                    sigmas=actual.tolist()))
        for key, bad in [('attention', 'unknown'), ('task', 'ref2va'), ('guidance_scale', 7.5),
                         ('dmd_denoising_steps', [1000, 750, 500, 250]),
                         ('dmd_denoising_steps', [999., 749., 500., 250.]),
                         ('video_shift', 3), ('audio_shift', 12), ('tile_size', 128), ('sparsity', .8)]:
            path.write_text(json.dumps({**manifest, key: bad}))
            assert not read() and config.kind == 0 and error.value
            records.append(dict(case=f'reject {key}={bad}', passed=True))
        for text in ('{', '{}', '[]', json.dumps(manifest) + 'garbage', 'x' * 65537):
            path.write_text(text)
            assert not read() and config.kind == 0
        records.append(dict(case='reject invalid manifest syntax/shape/size', passed=True))
        source = Path(__file__).resolve().parents[2] / 'cuda/prepare_fasth3.py'
        spec = importlib.util.spec_from_file_location('export_fasth3', source)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        filename, _ = exporter.write_rope_buffer(Path(folder), exporter.ARCH)
        saved = load_file(str(Path(folder) / filename))['rope.inv_freq']
        assert torch.equal(saved, MiniMaxH3RotaryPosEmbed(16, 10000.0).inv_freq)
        records.append(dict(case='exported derived RoPE buffer equals official reference', passed=True))
    args.output.write_text(json.dumps(dict(passed=True, cases=records), indent=2) + '\n')
    print(f'{len(records)} FastH3 contract checks passed', flush=True)


if __name__ == '__main__':
    main()
