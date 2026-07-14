#!/usr/bin/env python3
"""Strict, deterministic correctness gates for native EXAONE-4.5 W4A4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class Comparison:
    mean_abs_error: float
    max_abs_error: float
    token_agreement: float
    fallback_counts: dict[str, int]

    def to_report(self, *, layer_tolerance: float, logit_tolerance: float, **extra):
        return {
            **asdict(self),
            "tolerances": {
                "layer_max_abs_error": layer_tolerance,
                "logit_max_abs_error": logit_tolerance,
            },
            **extra,
        }


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    reference_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    counters: dict[str, int],
) -> Comparison:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"reference and candidate logits must have the same shape; got "
            f"{tuple(reference.shape)} and {tuple(candidate.shape)}"
        )
    if not reference_tokens or not candidate_tokens:
        raise ValueError("token sequences must be non-empty")
    if len(reference_tokens) != len(candidate_tokens):
        raise ValueError("token sequences must have the same length")
    delta = (reference.float() - candidate.float()).abs()
    return Comparison(
        mean_abs_error=delta.mean().item(),
        max_abs_error=delta.max().item(),
        token_agreement=sum(a == b for a, b in zip(reference_tokens, candidate_tokens))
        / len(reference_tokens),
        fallback_counts={str(k): int(v) for k, v in counters.items()},
    )


def require_local_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{label} local checkpoint path does not exist: {path}. "
            "This correctness harness never downloads or substitutes a model."
        )
    return path.resolve()


def _worker_counter_snapshot(_worker, reset: bool = False) -> dict[str, int]:
    from flatquant_vllm_plugin.w4a4_ops import dispatch_counters

    if reset:
        dispatch_counters.reset()
        return {}
    return dispatch_counters.snapshot()


def _worker_selection_snapshot(_worker) -> dict:
    from flatquant_vllm_plugin.w4a4_ops import selected_w4a4_projections
    prefixes = selected_w4a4_projections.snapshot()
    return {
        "w4a4_projection_count": len(prefixes),
        "w4a4_projection_prefixes": list(prefixes),
        "w4a16_fallback": 0,
        "bf16_fallback": 0,
    }


def query_worker_selection(llm) -> dict:
    snapshots = llm.collective_rpc(_worker_selection_snapshot)
    prefixes = sorted({prefix for item in snapshots for prefix in item["w4a4_projection_prefixes"]})
    return {
        "evidence_kind": "model_construction_selection",
        "w4a4_projection_count": len(prefixes),
        "w4a4_projection_prefixes": prefixes,
        "w4a16_fallback": sum(int(item["w4a16_fallback"]) for item in snapshots),
        "bf16_fallback": sum(int(item["bf16_fallback"]) for item in snapshots),
    }


def aggregate_worker_counters(snapshots: Sequence[dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for snapshot in snapshots:
        for name, value in snapshot.items():
            totals[name] = totals.get(name, 0) + int(value)
    if totals.get("w4a4", 0) <= 0:
        raise RuntimeError("W4A4 backend did not dispatch on any vLLM worker")
    if any(totals.get(name, 0) for name in ("w4a16_fallback", "bf16_fallback")):
        raise RuntimeError(f"strict W4A4 execution observed fallback: {totals}")
    return totals


def reset_worker_counters(llm) -> None:
    llm.collective_rpc(_worker_counter_snapshot, kwargs={"reset": True})


def query_worker_counters(llm) -> dict[str, int]:
    return aggregate_worker_counters(llm.collective_rpc(_worker_counter_snapshot))


def _assert_strict() -> None:
    if os.environ.get("FLATQUANT_W4A4_STRICT") != "1":
        raise RuntimeError("set FLATQUANT_W4A4_STRICT=1 for every correctness gate")


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _load_llm(model: Path, enforce_eager: bool):
    # Importing the plugin registers flatquant_w4a4 before vLLM reads config.json.
    import flatquant_vllm_plugin
    flatquant_vllm_plugin.register()
    from flatquant_vllm_plugin.w4a4_ops import selected_w4a4_projections
    selected_w4a4_projections.reset()
    from vllm import LLM

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=False)
    vocab_size = config.get_text_config().vocab_size
    return LLM(
        model=str(model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        trust_remote_code=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        skip_mm_profiling=True,
        kv_cache_memory_bytes=32 * 1024 * 1024,
        max_logprobs=vocab_size,
    )


def _require_flatquant_meta_kernels() -> list[str]:
    """Verify every CUDA op used by W4A4 has an abstract implementation."""
    import deploy._CUDA  # noqa: F401  # Register C++ schemas before querying dispatch.
    # Importing transform registers the two Python custom ops and fake kernels.
    from flatquant_vllm_plugin import transform as _transform  # noqa: F401
    from flatquant_vllm_plugin import w4a4_ops as _w4a4_ops  # noqa: F401

    names = (
        "flatquant::quantize_pack_i4",
        "flatquant::w4a4_linear",
        "flatquant::kron_transform",
        "flatquant::left_transform",
    )
    missing = [
        name for name in names
        if not torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta")
    ]
    if missing:
        raise RuntimeError(f"W4A4 CUDA Graph capture requires Meta kernels: {missing}")
    return list(names)


def run_cuda_graph_probe(
    model: str | Path,
    *,
    replay_cases: Sequence[dict],
) -> dict:
    """Exercise vLLM graph replay with equal shapes and changed token values.

    The probe deliberately uses the public vLLM engine rather than a raw
    ``torch.cuda.CUDAGraph`` around one kernel.  Thus the evidence covers the
    same piecewise graph/custom-op path used while serving.
    """
    model = require_local_path(model, "model")
    if len(replay_cases) < 2:
        raise ValueError("provide at least two CUDA Graph replay cases")
    if len({int(case["output_length"]) for case in replay_cases}) < 2:
        raise ValueError("CUDA Graph replay cases must cover two decode lengths")
    if len({len(case["first"]) for case in replay_cases}) < 2:
        raise ValueError("CUDA Graph replay cases must cover two prefill shapes")
    for case in replay_cases:
        first, second = case["first"], case["second"]
        if len(first) != len(second):
            raise ValueError("CUDA Graph prompt pairs must have the same shape")
        if first == second:
            raise ValueError("CUDA Graph prompt pairs must contain changed values")
    meta_kernels = _require_flatquant_meta_kernels()
    llm = _load_llm(model, False)
    selection_evidence = query_worker_selection(llm)
    if selection_evidence["w4a4_projection_count"] != 4:
        raise RuntimeError(
            "tiny conditional fixture must select exactly four fused W4A4 projections; "
            f"got {selection_evidence}"
        )
    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(model).get_text_config().vocab_size
    # Warm every shape/value before taking the allocator baseline. vLLM owns
    # graph pools and shape buckets; no benchmark-side workspace is allocated.
    replays = []
    for case in replay_cases:
        first, second = case["first"], case["second"]
        steps = int(case["output_length"])
        _candidate_logits(llm, first, steps, vocab_size, observe_counters=False)
        _candidate_logits(llm, second, steps, vocab_size, observe_counters=False)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        observations = []
        memory_samples = []
        for prompt in (first, second, first, second):
            logits, tokens, _ = _candidate_logits(
                llm, prompt, steps, vocab_size, observe_counters=False
            )
            torch.cuda.synchronize()
            observations.append((logits, tokens))
            memory_samples.append(torch.cuda.memory_allocated())
        responds = (
            not torch.equal(observations[0][0], observations[1][0])
            and torch.equal(observations[0][0], observations[2][0])
            and torch.equal(observations[1][0], observations[3][0])
        )
        stable = all(value == baseline for value in memory_samples)
        if not responds:
            raise RuntimeError("same-shape CUDA Graph replay ignored or corrupted changed values")
        if not stable:
            raise RuntimeError(
                f"CUDA Graph replay allocation changed: baseline={baseline}, samples={memory_samples}"
            )
        replays.append({
            "prompt_shape": [1, len(first)],
            "output_length": steps,
            "outputs_respond": True,
            "allocated_memory_stable": True,
            "allocated_memory_bytes": baseline,
            "replay_count": 4,
        })
    return {
        "command": "cuda_graph",
        "verified_fixture": "tiny_conditional_w4a4",
        "enforce_eager": False,
        "same_shape_changed_values": True,
        "allocated_memory_stable": True,
        "native_generation_completed": True,
        "replays": replays,
        "meta_kernels": meta_kernels,
        "selection_evidence": selection_evidence,
    }


def _dense_step_logits(step_logprobs, vocab_size: int) -> torch.Tensor:
    values = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    for token_id, item in step_logprobs.items():
        values[int(token_id)] = float(item.logprob)
    if not torch.isfinite(values).all():
        raise RuntimeError("vLLM did not return full-vocabulary logprobs; use logprobs=-1")
    return values


def _candidate_logits(
    llm, prompt_ids: list[int], steps: int, vocab_size: int, *, observe_counters=True
):
    from vllm import SamplingParams
    if observe_counters:
        reset_worker_counters(llm)
    result = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0, max_tokens=steps, ignore_eos=True, logprobs=-1),
    )[0].outputs[0]
    logits = torch.stack([_dense_step_logits(x, vocab_size) for x in result.logprobs])
    counters = query_worker_counters(llm) if observe_counters else None
    return logits, list(result.token_ids), counters


def _reference_logits(reference: Path, prompt_ids: list[int], steps: int):
    from benchmarks.exaone45.common import load_flatquant_model
    model, _ = load_flatquant_model(reference, "cuda", dtype="float16", eval_mode="deploy")
    tokens = list(prompt_ids)
    logits = []
    with torch.inference_mode():
        for _ in range(steps):
            output = model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False)
            # vLLM's public offline API exposes full-vocabulary logprobs, not
            # raw logits. Compare the equivalent normalized reference values.
            step = output.logits[0, -1].float().log_softmax(dim=-1).cpu()
            logits.append(step)
            tokens.append(int(step.argmax()))
    return torch.stack(logits), tokens[len(prompt_ids):]


def run_generate(args) -> dict:
    model = require_local_path(args.model, "model")
    _assert_strict()
    llm = _load_llm(model, args.enforce_eager)
    from vllm import SamplingParams

    reset_worker_counters(llm)
    outputs = llm.generate(
        [args.prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens)
    )
    token_ids = list(outputs[0].outputs[0].token_ids)
    if len(token_ids) != args.max_tokens:
        raise RuntimeError(
            f"expected {args.max_tokens} generated tokens, got {len(token_ids)}"
        )
    counters = query_worker_counters(llm)
    return {
        "command": "generate",
        "generated_token_ids": token_ids,
        "fallback_counts": counters,
        "strict": True,
        "w4a4_calls": counters["w4a4"],
    }


def run_logits(args) -> dict:
    _assert_strict()
    model = require_local_path(args.model, "model")
    reference = require_local_path(args.reference, "reference")
    if None in (args.layer_tolerance, args.logit_tolerance, args.min_token_agreement):
        raise RuntimeError("provide recorded --layer-tolerance, --logit-tolerance, and --min-token-agreement")
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    vocab_size = AutoConfig.from_pretrained(model).get_text_config().vocab_size
    llm = _load_llm(model, True)
    candidate, candidate_tokens, counters = _candidate_logits(llm, prompt_ids, args.max_tokens, vocab_size)
    reference_logits, reference_tokens = _reference_logits(reference, prompt_ids, args.max_tokens)
    comparison = compare_logits(reference_logits, candidate, reference_tokens, candidate_tokens, counters)
    if (comparison.mean_abs_error > args.layer_tolerance
            or comparison.max_abs_error > args.logit_tolerance
            or comparison.token_agreement < args.min_token_agreement):
        raise RuntimeError(f"logits gate failed: {comparison}")
    return comparison.to_report(
        layer_tolerance=args.layer_tolerance,
        logit_tolerance=args.logit_tolerance,
        command="logits",
        min_token_agreement=args.min_token_agreement,
    )


def run_ppl(args) -> dict:
    _assert_strict()
    model = require_local_path(args.model, "model")
    dataset = require_local_path(args.dataset_path, "WikiText-2 dataset")
    if args.ppl_min is None or args.ppl_max is None:
        raise RuntimeError("provide the recorded --ppl-min and --ppl-max")
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    ids = tokenizer.encode(dataset.read_text(), add_special_tokens=False)
    if len(ids) < 2:
        raise RuntimeError("WikiText-2 dataset must tokenize to at least two tokens")
    ids = ids[: args.max_tokens]
    llm = _load_llm(model, True)
    reset_worker_counters(llm)
    result = llm.generate(
        [{"prompt_token_ids": ids}],
        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1),
    )[0]
    losses = []
    for token_id, entry in zip(ids[1:], result.prompt_logprobs[1:]):
        if entry is None or token_id not in entry:
            raise RuntimeError("vLLM prompt_logprobs omitted a target token")
        losses.append(-float(entry[token_id].logprob))
    ppl = float(torch.tensor(losses).mean().exp())
    counters = query_worker_counters(llm)
    if not args.ppl_min <= ppl <= args.ppl_max:
        raise RuntimeError(f"PPL {ppl} outside recorded range [{args.ppl_min}, {args.ppl_max}]")
    return {"command": "ppl", "dataset": "wikitext2", "ppl": ppl,
            "ppl_range": [args.ppl_min, args.ppl_max], "fallback_counts": counters}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=32)
    generate.add_argument("--enforce-eager", action="store_true")
    generate.add_argument("--report", type=Path, default=Path("w4a4-generate.json"))
    generate.set_defaults(func=run_generate)
    logits = sub.add_parser("logits")
    logits.add_argument("--model", required=True)
    logits.add_argument("--reference", required=True)
    logits.add_argument("--prompt", default="Explain activation quantization in one paragraph.")
    logits.add_argument("--max-tokens", type=int, default=8)
    logits.add_argument("--layer-tolerance", type=float)
    logits.add_argument("--logit-tolerance", type=float)
    logits.add_argument("--min-token-agreement", type=float)
    logits.add_argument("--report", type=Path, default=Path("w4a4-logits.json"))
    logits.set_defaults(func=run_logits)
    ppl = sub.add_parser("ppl")
    ppl.add_argument("--model", required=True)
    ppl.add_argument("--dataset", choices=["wikitext2"], required=True)
    ppl.add_argument("--dataset-path", default="wikitext-2-test.txt")
    ppl.add_argument("--max-tokens", type=int, default=2048)
    ppl.add_argument("--ppl-min", type=float)
    ppl.add_argument("--ppl-max", type=float)
    ppl.add_argument("--report", type=Path, default=Path("w4a4-ppl.json"))
    ppl.set_defaults(func=run_ppl)
    graph = sub.add_parser("cuda-graph")
    graph.add_argument("--model", required=True)
    graph.add_argument("--first-prompt-ids", type=int, nargs="+", default=[1, 3])
    graph.add_argument("--second-prompt-ids", type=int, nargs="+", default=[1, 4])
    graph.add_argument("--report", type=Path, default=Path("w4a4-cuda-graph.json"))
    graph.set_defaults(
        func=lambda args: run_cuda_graph_probe(
            args.model,
            replay_cases=[
                {"first": args.first_prompt_ids, "second": args.second_prompt_ids,
                 "output_length": 1},
                {"first": args.first_prompt_ids + [3],
                 "second": args.second_prompt_ids + [4], "output_length": 2},
            ],
        )
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = args.func(args)
        _write_report(args.report, payload)
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
