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


def _assert_strict() -> None:
    if os.environ.get("FLATQUANT_W4A4_STRICT") != "1":
        raise RuntimeError("set FLATQUANT_W4A4_STRICT=1 for every correctness gate")


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _load_llm(model: Path, enforce_eager: bool):
    # Importing the plugin registers flatquant_w4a4 before vLLM reads config.json.
    import flatquant_vllm_plugin  # noqa: F401
    from vllm import LLM

    return LLM(
        model=str(model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        trust_remote_code=False,
    )


def run_generate(args) -> dict:
    model = require_local_path(args.model, "model")
    _assert_strict()
    llm = _load_llm(model, args.enforce_eager)
    from vllm import SamplingParams

    outputs = llm.generate(
        [args.prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens)
    )
    token_ids = list(outputs[0].outputs[0].token_ids)
    if len(token_ids) != args.max_tokens:
        raise RuntimeError(
            f"expected {args.max_tokens} generated tokens, got {len(token_ids)}"
        )
    # vLLM workers are separate processes. Until the plugin exposes worker RPC
    # counters, successful strict execution proves no fallback (there is no
    # fallback branch); it cannot honestly report the worker's W4A4 call total.
    return {
        "command": "generate",
        "generated_token_ids": token_ids,
        "fallback_counts": {"fallback": 0},
        "strict": True,
        "w4a4_calls": None,
    }


def run_logits(args) -> dict:
    _assert_strict()
    require_local_path(args.model, "model")
    require_local_path(args.reference, "reference")
    raise RuntimeError(
        "logits gate requires the real-model reference adapter and artifacts; "
        "they are not bundled with this repository"
    )


def run_ppl(args) -> dict:
    _assert_strict()
    require_local_path(args.model, "model")
    raise RuntimeError(
        "ppl gate requires the real WikiText-2 tokenization/reference artifacts; "
        "they are not bundled with this repository"
    )


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
    logits.add_argument("--report", type=Path, default=Path("w4a4-logits.json"))
    logits.set_defaults(func=run_logits)
    ppl = sub.add_parser("ppl")
    ppl.add_argument("--model", required=True)
    ppl.add_argument("--dataset", choices=["wikitext2"], required=True)
    ppl.add_argument("--report", type=Path, default=Path("w4a4-ppl.json"))
    ppl.set_defaults(func=run_ppl)
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
