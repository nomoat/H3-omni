"""Deterministic, source-level CUDA candidates for the H3 correctness seed.

This module edits only a supplied copy of cuda/h3_gpu_cuda.cu.  It does not
modify the evaluator, precision policy, weights, step count, or layer count.
Every candidate must still pass the external frozen numerical validator.

The first campaign intentionally leaves cuBLAS algorithms and workspace size
unchanged.  Its axes change allocation lifetime, descriptor lifetime, and
launch geometry without changing floating-point expressions or reduction
geometry.  Optional Lt workspace/algorithm axes are exposed for later runs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


DEFAULT = {
    "elementwise_threads": 256,
    "scratch_mask": 0,
    "lt_cache": False,
    "lt_workspace_mib": 1,
    "lt_algorithm_rank": 0,
}

# Bits: MLP fc1/activation; final-head tensors; attention LSE/FP32 workspace;
# patch projections, their nested scatter intermediate, and BF16 bias scratch.
SCRATCH_FAMILIES = ("mlp", "final_heads", "attention", "projections")


def normalize(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) - set(DEFAULT):
        raise ValueError(f"Unknown configuration fields: {set(config) - set(DEFAULT)}")
    result = DEFAULT | config
    for key in DEFAULT:
        if type(result[key]) is not type(DEFAULT[key]):
            raise ValueError(f"Wrong type for {key}: {result[key]!r}")
    if result["elementwise_threads"] not in (64, 128, 256, 512):
        raise ValueError("elementwise_threads must be 64, 128, 256, or 512")
    if not 0 <= result["scratch_mask"] < 16:
        raise ValueError("scratch_mask must be between 0 and 15")
    if result["lt_workspace_mib"] not in (1, 4, 16):
        raise ValueError("lt_workspace_mib must be 1, 4, or 16")
    if result["lt_algorithm_rank"] not in (0, 1, 2):
        raise ValueError("lt_algorithm_rank must be 0, 1, or 2")
    return result


def replace_once(source: str, before: str, after: str) -> str:
    count = source.count(before)
    if count != 1:
        raise ValueError(f"Seed anchor occurs {count} times, expected one: {before[:100]!r}")
    return source.replace(before, after, 1)


def replace_function(source: str, signature: str, mutate) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    # Target seed functions contain no braces in strings or comments.
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"Unbalanced function: {signature}")
    old = source[start:end]
    new = mutate(old)
    if old == new:
        raise ValueError(f"Mutation made no change: {signature}")
    return source[:start] + new + source[end:]


LT_PLAN = r'''
// Source-generated experiment: cache only immutable Lt shape/alignment plans.
struct H3ResearchLtPlan {
    uint rows{}, k{}, n{};
    uint32_t alignment[5]{};
    cublasLtMatmulDesc_t descriptor{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulHeuristicResult_t algorithm{};
    ~H3ResearchLtPlan() {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (descriptor) cublasLtMatmulDescDestroy(descriptor);
    }
};
'''


LT_CACHED_FUNCTION = r'''static int linear_bias_f32(h3_gpu *g, h3_gpu_tensor *y, const h3_gpu_tensor *x,
                           const h3_gpu_tensor *w, const h3_gpu_tensor *bias,
                           uint rows, uint k, uint n) {
    constexpr size_t workspace_bytes = @WORKSPACE@ * 1024 * 1024;
    if (!g->lt_workspace && !cuda_ok(g, cudaMalloc(&g->lt_workspace, workspace_bytes))) return 0;
    // C and D both use y; bias is a separate epilogue input.  Keep its
    // alignment in the cache key without mislabeling it as D's alignment.
    const void *pointers[] = {w->data, x->data, y->data, y->data, bias->data};
    uint32_t alignment[5];
    for (int i = 0; i < 5; ++i) {
        alignment[i] = 256;
        while (reinterpret_cast<uintptr_t>(pointers[i]) % alignment[i]) alignment[i] /= 2;
    }
    H3ResearchLtPlan *plan = nullptr;
    for (auto *candidate : g->research_lt_plans) {
        if (candidate->rows == rows && candidate->k == k && candidate->n == n &&
            !memcmp(candidate->alignment, alignment, sizeof(alignment))) {
            plan = candidate;
            break;
        }
    }
    auto handle = reinterpret_cast<cublasLtHandle_t>(g->blas);
    if (!plan) {
        plan = new H3ResearchLtPlan;
        plan->rows = rows; plan->k = k; plan->n = n;
        memcpy(plan->alignment, alignment, sizeof(alignment));
        cublasLtMatmulPreference_t preference{};
        cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        int ok = blas_ok(g, cublasLtMatmulDescCreate(&plan->descriptor, CUBLAS_COMPUTE_32F, CUDA_R_32F)) &&
            blas_ok(g, cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose))) &&
            blas_ok(g, cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal))) &&
            blas_ok(g, cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) &&
            blas_ok(g, cublasLtMatmulDescSetAttribute(plan->descriptor, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias->data, sizeof(bias->data))) &&
            blas_ok(g, cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_32F, k, n, k)) &&
            blas_ok(g, cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_32F, k, rows, k)) &&
            blas_ok(g, cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_32F, n, rows, n)) &&
            blas_ok(g, cublasLtMatmulPreferenceCreate(&preference)) &&
            blas_ok(g, cublasLtMatmulPreferenceSetAttribute(preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
        const cublasLtMatmulPreferenceAttributes_t attributes[] = {
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
        for (int i = 0; i < 4 && ok; ++i)
            ok = blas_ok(g, cublasLtMatmulPreferenceSetAttribute(preference, attributes[i], &alignment[i], sizeof(alignment[i])));
        constexpr int requested = @RANK@ + 1;
        cublasLtMatmulHeuristicResult_t algorithms[requested]{};
        int count = 0;
        if (ok) ok = blas_ok(g, cublasLtMatmulAlgoGetHeuristic(handle, plan->descriptor,
            plan->a, plan->b, plan->c, plan->c, preference, requested, algorithms, &count));
        if (ok && count < requested) {
            h3_gpu_set_error(g, "Requested cuBLASLt FP32 bias algorithm rank is unavailable");
            ok = 0;
        }
        if (ok && (algorithms[@RANK@].state != CUBLAS_STATUS_SUCCESS ||
                   algorithms[@RANK@].workspaceSize > workspace_bytes)) {
            h3_gpu_set_error(g, "cuBLASLt heuristic returned an unusable algorithm");
            ok = 0;
        }
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (!ok) { delete plan; return 0; }
        plan->algorithm = algorithms[@RANK@];
        g->research_lt_plans.push_back(plan);
    }
    // Bias tensors differ between layers even when their shapes are identical.
    int ok = blas_ok(g, cublasLtMatmulDescSetAttribute(plan->descriptor,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias->data, sizeof(bias->data)));
    float alpha = 1, beta = 0;
    if (ok) ok = blas_ok(g, cublasLtMatmul(handle, plan->descriptor, &alpha, w->data,
        plan->a, x->data, plan->b, &beta, y->data, plan->c, y->data, plan->c,
        &plan->algorithm.algo, g->lt_workspace, workspace_bytes, g->stream));
    ++g->stats.mps_linear_dispatches;
    return ok;
}'''


SCRATCH_HELPER = r'''
// Slots are per h3_gpu/stream and never shared by nested operation families.
// Tensor contents are overwritten by their original operations before reading.
static h3_gpu_tensor *research_scratch(h3_gpu *g, unsigned slot, size_t n, h3_gpu_dtype d) {
    if (!g || slot >= 12 || n > SIZE_MAX / item_size(d)) return nullptr;
    auto *t = g->research_scratch[slot];
    if (t && (t->dtype != d || t->elements < n)) {
        h3_gpu_tensor_free(t);
        g->research_scratch[slot] = nullptr;
        t = nullptr;
    }
    if (!t) {
        t = allocate(g, n, d);
        g->research_scratch[slot] = t;
    }
    return t;
}
'''


def scratch_mutations(source: str, mask: int) -> str:
    source = replace_once(source, "    void *lt_workspace{};",
                          "    void *lt_workspace{};\n    h3_gpu_tensor *research_scratch[12]{};")
    source = replace_once(source, "    if (g->dnn)\n        cudnnDestroy(g->dnn);",
                          "    for (auto *tensor : g->research_scratch) h3_gpu_tensor_free(tensor);\n"
                          "    if (g->dnn)\n        cudnnDestroy(g->dnn);")
    source = replace_once(source, 'extern "C" h3_gpu_tensor *h3_gpu_tensor_new_f32',
                          SCRATCH_HELPER + '\nextern "C" h3_gpu_tensor *h3_gpu_tensor_new_f32')

    def edits(signature, replacements):
        nonlocal source
        def apply(text):
            for old, new in replacements:
                text = replace_once(text, old, new)
            return text
        source = replace_function(source, signature, apply)

    if mask & 1:
        edits('extern "C" int h3_gpu_mlp_bf16(', [
            ("allocate(g, (size_t)rows * hidden * 2, H3_GPU_BF16)", "research_scratch(g, 0, (size_t)rows * hidden * 2, H3_GPU_BF16)"),
            ("allocate(g, (size_t)rows * hidden, H3_GPU_BF16)", "research_scratch(g, 1, (size_t)rows * hidden, H3_GPU_BF16)"),
            ("    h3_gpu_tensor_free(fc1);\n    h3_gpu_tensor_free(act);", "    // Slots 0/1 survive until this GPU context is destroyed."),
        ])
    if mask & 2:
        edits('extern "C" int h3_gpu_final_heads_f32(', [
            ("allocate(g, rows, H3_GPU_U32)", "research_scratch(g, 2, rows, H3_GPU_U32)"),
            ("allocate(g, size_t(rows) * width, H3_GPU_BF16)", "research_scratch(g, 3, size_t(rows) * width, H3_GPU_BF16)"),
            ("allocate(g, size_t(rows) * width, H3_GPU_F32)", "research_scratch(g, 4, size_t(rows) * width, H3_GPU_F32)"),
            ("allocate(g, size_t(rows) * 96, H3_GPU_F32)", "research_scratch(g, 5, size_t(rows) * 96, H3_GPU_F32)"),
            ("allocate(g, size_t(rows) * 32, H3_GPU_F32)", "research_scratch(g, 6, size_t(rows) * 32, H3_GPU_F32)"),
            ("    for (auto *tensor : {map, normalized, f32, video, audio}) h3_gpu_tensor_free(tensor);", "    // Slots 2..6 survive until this GPU context is destroyed."),
        ])
    if mask & 4:
        edits("static int sdpa(", [
            ("allocate(g, size_t(batch) * h * n, H3_GPU_F32)", "research_scratch(g, 7, size_t(batch) * h * n, H3_GPU_F32)"),
            ("        h3_gpu_tensor_free(lse);", "        // Slot 7 retains the FlashAttention log-sum-exp workspace."),
            ("allocate(g, size * 4 + (size_t)bh * chunk * n, H3_GPU_F32)", "research_scratch(g, 8, size * 4 + (size_t)bh * chunk * n, H3_GPU_F32)"),
            ("    h3_gpu_tensor_free(buf);", "    // Slot 8 retains the separate FP32 attention workspace."),
        ])
    if mask & 8:
        edits('extern "C" int h3_gpu_patch_linear_bf16(', [
            ("allocate(g, (size_t)r * n, H3_GPU_F32)", "research_scratch(g, 9, (size_t)r * n, H3_GPU_F32)"),
            ("    h3_gpu_tensor_free(t);", "    // Slot 9 is distinct from the calling scatter projection slot."),
        ])
        edits('extern "C" int h3_gpu_patch_linear_bf16_map(', [
            ("allocate(g, (size_t)r * n, H3_GPU_BF16)", "research_scratch(g, 10, (size_t)r * n, H3_GPU_BF16)"),
            ("    h3_gpu_tensor_free(t);", "    // Slot 10 is distinct from the nested FP32 projection slot."),
        ])
        edits("static int linear(", [
            ("allocate(g, size_t(rows) * n, H3_GPU_F32)", "research_scratch(g, 11, size_t(rows) * n, H3_GPU_F32)"),
            ("        h3_gpu_tensor_free(acc);", "        // Slot 11 preserves the original FP32 accumulation and BF16 cast."),
        ])
    return source


def generate(source_text: str, config: dict[str, Any]) -> str:
    """Return a complete candidate from the immutable seed, never from a parent.

    Repeated calls with the same seed/config return identical source.  A caller
    may compare this output against any branch parent's generated source.
    """
    config = normalize(config)
    if "// H3_AUTORESEARCH_GENERATED " in source_text:
        raise ValueError("Input is already generated; use the immutable seed")
    source = source_text
    threads = config["elementwise_threads"]
    if threads != 256:
        source = replace_once(source,
            "return dispatch(g, n, dim3((x + 255) / 256), dim3(256), f);",
            f"return dispatch(g, n, dim3((x + {threads - 1}) / {threads}), dim3({threads}), f);")
    if config["scratch_mask"]:
        source = scratch_mutations(source, config["scratch_mask"])
    if config["lt_cache"]:
        source = replace_once(source, "struct h3_gpu {", LT_PLAN + "\nstruct h3_gpu {")
        source = replace_once(source, "    void *lt_workspace{};",
            "    void *lt_workspace{};\n    std::vector<H3ResearchLtPlan *> research_lt_plans;")
        source = replace_once(source, "    if (g->blas)\n        cublasDestroy(g->blas);",
            "    for (auto *plan : g->research_lt_plans) delete plan;\n"
            "    if (g->blas)\n        cublasDestroy(g->blas);")
        replacement = LT_CACHED_FUNCTION.replace("@WORKSPACE@", str(config["lt_workspace_mib"]))
        replacement = replacement.replace("@RANK@", str(config["lt_algorithm_rank"]))
        source = replace_function(source, "static int linear_bias_f32(", lambda _: replacement)
    else:
        if config["lt_workspace_mib"] != 1:
            source = replace_once(source, "constexpr size_t workspace_bytes = 1024 * 1024;",
                f'constexpr size_t workspace_bytes = {config["lt_workspace_mib"]} * 1024 * 1024;')
        rank = config["lt_algorithm_rank"]
        if rank:
            source = replace_once(source, "    cublasLtMatmulHeuristicResult_t algorithm{};",
                f"    cublasLtMatmulHeuristicResult_t algorithms[{rank + 1}]{{}};")
            source = replace_once(source, "preference, 1, &algorithm, &count)",
                f"preference, {rank + 1}, algorithms, &count)")
            source = replace_once(source, 'if (ok && !count) { h3_gpu_set_error(g, "No cuBLASLt FP32 bias algorithm"); ok = 0; }',
                f'if (ok && count < {rank + 1}) {{ h3_gpu_set_error(g, "Requested cuBLASLt FP32 bias algorithm rank is unavailable"); ok = 0; }}')
            source = replace_once(source, "&algorithm.algo, g->lt_workspace", f"&algorithms[{rank}].algo, g->lt_workspace")
    if config == DEFAULT:
        if source != source_text:
            raise AssertionError("Baseline changed")
        return source
    if source == source_text:
        raise AssertionError("Non-baseline candidate made no change")
    marker = "// H3_AUTORESEARCH_GENERATED " + json.dumps(config, sort_keys=True, separators=(",", ":"))
    return marker + "\n" + source


def families(config: dict[str, Any]) -> list[str]:
    config = normalize(config)
    result = [name for bit, name in enumerate(SCRATCH_FAMILIES) if config["scratch_mask"] & (1 << bit)]
    if config["lt_cache"]:
        result.append("lt_cache")
    if config["elementwise_threads"] != 256:
        result.append("elementwise_launch")
    if config["lt_workspace_mib"] != 1 or config["lt_algorithm_rank"]:
        result.append("lt_algorithm")
    return result


def schedule(count: int = 100, source_text: str | None = None) -> list[dict[str, Any]]:
    """Prioritize isolated mutations, then branch combinations; 127 real configs.

    Parent is the nearest earlier configuration after removing one axis.  The
    controller may instead retain another successful parent; generate() always
    reconstructs the full declared configuration from the original seed.
    """
    if not 1 <= count <= 127:
        raise ValueError("The stable campaign has 127 non-baseline configurations")
    configs = [normalize({"scratch_mask": mask, "lt_cache": cache, "elementwise_threads": threads})
               for mask, cache, threads in itertools.product(range(16), (False, True), (256, 128, 512, 64))]
    configs = [cfg for cfg in configs if cfg != DEFAULT]
    def order(cfg):
        bits = cfg["scratch_mask"].bit_count()
        launch = cfg["elementwise_threads"] != 256
        return (bits + int(cfg["lt_cache"]) + int(launch),
                (256, 128, 512, 64).index(cfg["elementwise_threads"]),
                -int(cfg["lt_cache"]), cfg["scratch_mask"])
    configs.sort(key=order)
    # After isolated axes, exercise a concrete allocation/cache branch through
    # all families.  A pure low-order factorial prefix would omit the combined
    # plan most likely to expose interactions and retained-memory limits.
    singles = [cfg for cfg in configs if order(cfg)[0] == 1]
    interaction_branch = [normalize({"scratch_mask": mask, "lt_cache": True})
                          for mask in (4, 5, 7, 15)]
    interaction_branch += [normalize({"scratch_mask": 15, "lt_cache": True,
                                      "elementwise_threads": threads})
                           for threads in (128, 512, 64)]
    priority = singles + interaction_branch
    configs = priority + [cfg for cfg in configs if cfg not in priority]
    configs = configs[:count]
    ids = {json.dumps(DEFAULT, sort_keys=True): "baseline"}
    output = []
    hashes = set()
    for i, cfg in enumerate(configs, 1):
        parent_options = []
        if cfg["elementwise_threads"] != 256:
            parent_options.append(cfg | {"elementwise_threads": 256})
        for bit in range(4):
            if cfg["scratch_mask"] & (1 << bit):
                parent_options.append(cfg | {"scratch_mask": cfg["scratch_mask"] & ~(1 << bit)})
        if cfg["lt_cache"]:
            parent_options.append(cfg | {"lt_cache": False})
        parent = next((ids[json.dumps(p, sort_keys=True)] for p in parent_options
                       if json.dumps(p, sort_keys=True) in ids), "baseline")
        record = {"id": f"candidate-{i:03d}", "parent": parent,
                  "config": cfg, "families": families(cfg)}
        if source_text is not None:
            generated = generate(source_text, cfg)
            # Exclude provenance comments from de-duplication.
            payload = generated.partition("\n")[2]
            digest = hashlib.sha256(payload.encode()).hexdigest()
            if digest in hashes:
                raise AssertionError("Duplicate executable-source mutation")
            hashes.add(digest)
            record["source_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
            record["mutation_sha256"] = digest
        ids[json.dumps(cfg, sort_keys=True)] = record["id"]
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("schedule")
    plan.add_argument("--seed", type=Path, required=True)
    plan.add_argument("--count", type=int, default=100)
    plan.add_argument("--output", type=Path)
    apply = commands.add_parser("apply")
    apply.add_argument("--seed", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--config-json", required=True)
    args = parser.parse_args()
    seed = args.seed.read_text()
    if args.command == "schedule":
        result = {"seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
                  "baseline": DEFAULT, "candidates": schedule(args.count, seed)}
        text = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end="")
    else:
        if args.seed.resolve() == args.output.resolve():
            raise ValueError("The immutable seed and output must be different files")
        args.output.write_text(generate(seed, json.loads(args.config_json)))


if __name__ == "__main__":
    main()
