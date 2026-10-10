"""Exact tokenizer tests and real 50-layer Qwen3-VL conditioning parity."""
import argparse
import ctypes as C
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Diagnostic intermediate snapshots")
    parser.add_argument("--prompt", action="append", default=[],
                        help="Additional prompt to check with both full text encoders; repeatable")
    args = parser.parse_args()
    root, output = Path(args.weights), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lib = C.CDLL(str(Path("build-cuda/libh3_tokenizer.so").resolve()))
    ptr, size = C.c_void_p, C.c_size_t
    lib.h3_tokenizer_load.argtypes, lib.h3_tokenizer_load.restype = [C.c_char_p, ptr, size], ptr
    lib.h3_tokenizer_encode.argtypes = [ptr, C.c_char_p, C.c_int, C.POINTER(ptr), C.POINTER(size), ptr, size]
    lib.h3_tokenizer_encode.restype = C.c_int
    lib.h3_tokenizer_ids_free.argtypes = [ptr]
    lib.h3_tokenizer_free.argtypes = [ptr]
    error = C.create_string_buffer(1024)
    tokenizer_file = root / "tokenizer/tokenizer.json"
    tokenizer = lib.h3_tokenizer_load(str(tokenizer_file).encode(), error, len(error))
    if not tokenizer:
        raise RuntimeError(error.value.decode())
    reference_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    prompts = ["", "A cat walks through a sunny garden.", "一只猫在花园里走动。", "a\r\nb\n\n c\t ",
               "café cafe\u0301", "I'm WE'RE can't he'd", "1234567890 ½²Ⅷ", "🙂👨‍👩‍👧‍👦", "<|im_start|>user\nhello<|im_end|>"]
    ids_by_prompt, records = {}, []
    for prompt in dict.fromkeys(prompts + args.prompt):
        ids, count = ptr(), size()
        ok = lib.h3_tokenizer_encode(tokenizer, prompt.encode(), 0, C.byref(ids), C.byref(count), error, len(error))
        if not ok:
            raise RuntimeError(error.value.decode())
        native_ids = list(C.cast(ids, C.POINTER(C.c_uint32))[:count.value])
        lib.h3_tokenizer_ids_free(ids)
        expected = reference_tokenizer.encode(prompt, add_special_tokens=False).ids
        row = dict(prompt=prompt, passed=native_ids == expected, native=native_ids, reference=expected)
        records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        ids_by_prompt[prompt] = native_ids
    lib.h3_tokenizer_free(tokenizer)
    if not all(r["passed"] for r in records):
        (output / "tokenizer-results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
        raise RuntimeError("Tokenizer IDs differ")
    if not args.tokenizer_only:
        native = C.CDLL(str(Path("build-cuda/libh3_model.so").resolve()))
        native.h3_phase1_text.argtypes = [C.c_char_p, ptr, size, ptr, ptr, size]
        native.h3_phase1_text.restype = C.c_int
        started = time.monotonic()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(root / "processor")
        print("Qwen reference loaded", time.monotonic() - started, flush=True)
        limit = json.loads(Path(__file__).with_name("acceptance.json").read_text())["model"]["text_embedding"]
        for i, prompt in enumerate(prompts[1:3] + args.prompt):
            traces, hooks = {}, []
            if args.trace:
                trace_dir = (output / f'trace-{i}').resolve()
                trace_dir.mkdir(exist_ok=True)
                os.environ['H3_CUDA_TEXT_TRACE_DIR'] = str(trace_dir)
                def capture(label, before=False):
                    def hook(module, inputs, result=None):
                        value = inputs[0] if before else result
                        traces[label] = value.detach().cpu().contiguous()
                    return hook
                layers = model.model.language_model.layers
                for index, layer in enumerate(layers[:50]):
                    hooks.append(layer.register_forward_hook(capture(f'block-{index:02d}')))
                first = layers[0]
                for label, module in (
                    ('norm1', first.input_layernorm), ('norm2', first.post_attention_layernorm),
                    ('query_projection', first.self_attn.q_proj), ('key_projection', first.self_attn.k_proj),
                    ('value_projection', first.self_attn.v_proj), ('query_norm', first.self_attn.q_norm),
                    ('key_norm', first.self_attn.k_norm), ('attention_output', first.self_attn.o_proj),
                    ('mlp_gate', first.mlp.gate_proj), ('mlp_up', first.mlp.up_proj),
                    ('mlp_output', first.mlp.down_proj)):
                    hooks.append(module.register_forward_hook(capture(label)))
                hooks.append(first.self_attn.o_proj.register_forward_pre_hook(capture('attention_heads', True)))
                hooks.append(first.mlp.down_proj.register_forward_pre_hook(capture('mlp_activation', True)))
            else:
                os.environ.pop('H3_CUDA_TEXT_TRACE_DIR', None)
            ids = torch.tensor(ids_by_prompt[prompt], dtype=torch.int32)
            got = torch.empty(1, ids.numel(), 5120, dtype=torch.bfloat16)
            started = time.monotonic()
            ok = native.h3_phase1_text(str(root / "text_encoder").encode(), ids.data_ptr(), ids.numel(), got.data_ptr(), error, len(error))
            if not ok:
                raise RuntimeError(error.value.decode())
            native_seconds = time.monotonic() - started
            with torch.inference_mode():
                expected = get_qwen3vl_prompt_embeds(model, processor, ids.tolist(), device=torch.device("cuda"), dtype=torch.bfloat16).cpu()
            if args.trace:
                for hook in hooks:
                    hook.remove()
                save_file(traces, str(trace_dir / 'reference.safetensors'))
                differences = []
                for name, reference in traces.items():
                    got_trace = torch.frombuffer(bytearray((trace_dir / (name + '.bf16')).read_bytes()),
                                                 dtype=torch.bfloat16).reshape(reference.shape).float()
                    delta_trace = got_trace - reference.float()
                    differences.append(dict(stage=name, relative_l2=float(delta_trace.norm()/reference.float().norm().clamp_min(1e-12)),
                                            unequal_fraction=float((got_trace != reference.float()).float().mean())))
                (trace_dir / 'comparison.json').write_text(json.dumps(differences, indent=2)+'\n')
                print(json.dumps({'trace':i,'differences':differences}), flush=True)
            delta = got.float() - expected.float()
            rel = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
            maximum = float(delta.abs().max() / expected.float().abs().max().clamp_min(1e-12))
            passed = bool(torch.isfinite(got).all() and torch.isfinite(expected).all() and
                          rel <= limit["relative_l2"] and maximum <= limit["normalized_max"])
            row = dict(prompt=prompt, tokens=ids.numel(), relative_l2=rel, normalized_max=maximum, passed=passed,
                       native_load_and_encode_seconds=native_seconds)
            records.append(row)
            print(json.dumps(row), flush=True)
            save_file({"text": got, "reference_text": expected}, str(output / f"text-{i}.safetensors"))
    report = dict(passed=all(r["passed"] for r in records), cases=records, trace_enabled=args.trace)
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
