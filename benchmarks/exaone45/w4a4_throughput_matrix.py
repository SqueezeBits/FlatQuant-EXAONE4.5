#!/usr/bin/env python3
"""Controlled offline BF16/W4A16/W4A4 vLLM throughput matrix.

All three checkpoint arguments must be real local artifacts.  In particular,
this tool never substitutes the tiny graph fixture for the 33B W4A4 model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time


@dataclass(frozen=True)
class MatrixConfig:
    input_lengths: tuple[int, ...]
    concurrencies: tuple[int, ...]
    output_length: int

    def cases(self):
        for input_length in self.input_lengths:
            for concurrency in self.concurrencies:
                yield input_length, concurrency


def validate_model_paths(bf16, w4a16, w4a4) -> dict[str, Path]:
    paths = {name: Path(value).expanduser() for name, value in (
        ("bf16", bf16), ("w4a16", w4a16), ("w4a4", w4a4)
    )}
    for name in ("bf16", "w4a16"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"{name} local model directory does not exist: {paths[name]}")
    if not paths["w4a4"].is_dir():
        raise FileNotFoundError(
            f"real W4A4 local model directory does not exist: {paths['w4a4']}; "
            "the BF16/W4A16/W4A4 matrix was not run"
        )
    manifest = paths["w4a4"] / "flatquant_w4a4_config.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"real W4A4 artifact is missing {manifest.name}; the matrix was not run"
        )
    return {name: path.resolve() for name, path in paths.items()}


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata():
    import torch
    result = {"packages": {name: _version(name) for name in ("torch", "vllm", "triton")},
              "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                         "compute_capability": f"{props.major}.{props.minor}"}
    return result


def _prompt_ids(tokenizer, length):
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    if not ids:
        raise RuntimeError("tokenizer produced no seed tokens")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def _build_llm(backend, model, args):
    if backend == "w4a4":
        import flatquant_vllm_plugin
        flatquant_vllm_plugin.register()
    from vllm import LLM
    kwargs = dict(model=str(model), dtype="bfloat16", tensor_parallel_size=1,
                  enforce_eager=False, trust_remote_code=False,
                  max_model_len=max(args.input_lengths) + args.output_length,
                  kv_cache_dtype=args.kv_cache_dtype,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  disable_log_stats=False)
    if backend == "w4a4":
        kwargs["quantization"] = "flatquant_w4a4"
    return LLM(**kwargs)


def _run_backend(backend, model, config, args):
    import torch
    from vllm import SamplingParams
    from benchmarks.exaone45.vllm_w4a4 import query_worker_counters, reset_worker_counters
    llm = _build_llm(backend, model, args)
    tokenizer = llm.get_tokenizer()
    rows = []
    for input_length, concurrency in config.cases():
        prompts = [{"prompt_token_ids": _prompt_ids(tokenizer, input_length)}] * concurrency
        params = SamplingParams(temperature=0, max_tokens=config.output_length, ignore_eos=True)
        llm.generate(prompts, params, use_tqdm=False)  # warm shape bucket
        if backend == "w4a4":
            reset_worker_counters(llm)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        errors = []
        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        except Exception as error:
            outputs = []
            errors.append(f"{type(error).__name__}: {error}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        ttfts = [float(output.metrics.first_token_latency) for output in outputs
                 if getattr(output, "metrics", None) is not None
                 and output.metrics.first_token_latency is not None]
        counters = query_worker_counters(llm) if backend == "w4a4" and outputs else {}
        rows.append({
            "backend": backend, "input_length": input_length, "concurrency": concurrency,
            "output_length": config.output_length,
            "prompt_tokens_per_s": input_length * len(outputs) / elapsed,
            "requests_per_s": len(outputs) / elapsed,
            "ttft_median_s": statistics.median(ttfts) if ttfts else None,
            "ttft_p95_s": _percentile(ttfts, .95) if ttfts else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "completed_requests": len(outputs), "errors": errors,
            "dispatch_counters": counters,
        })
    del llm
    torch.cuda.empty_cache()
    return rows


def render_markdown(payload):
    lines = [f"# EXAONE throughput matrix ({payload['scope']})", "",
             f"Status: `{payload['status']}`", "",
             "| backend | input | concurrency | prompt tokens/s | requests/s | TTFT median | TTFT p95 | peak bytes | completed | errors | dispatch counters |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in payload["rows"]:
        counters = json.dumps(row["dispatch_counters"], sort_keys=True)
        display = {**row, "errors": "; ".join(row["errors"]) or "none", "counters": counters}
        lines.append("| {backend} | {input_length} | {concurrency} | {prompt_tokens_per_s:.3f} | {requests_per_s:.4f} | {ttft_median_s} | {ttft_p95_s} | {peak_gpu_memory_bytes} | {completed_requests} | {errors} | {counters} |".format(
            **display))
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-model", required=True)
    parser.add_argument("--w4a16-model", required=True)
    parser.add_argument("--w4a4-model", required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.9)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        paths = validate_model_paths(args.bf16_model, args.w4a16_model, args.w4a4_model)
        if os.environ.get("FLATQUANT_W4A4_STRICT") != "1":
            raise RuntimeError("set FLATQUANT_W4A4_STRICT=1 for the throughput matrix")
        config = MatrixConfig(tuple(args.input_lengths), tuple(args.concurrencies), args.output_length)
        rows = []
        for backend in ("bf16", "w4a16", "w4a4"):
            rows.extend(_run_backend(backend, paths[backend], config, args))
        payload = {"status": "verified", "scope": "real_33b_matrix", "environment": _metadata(),
                   "configuration": vars(config), "model_paths": {k: str(v) for k, v in paths.items()}, "rows": rows}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        markdown = args.markdown or args.json.with_suffix(".md")
        markdown.write_text(render_markdown(payload))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
