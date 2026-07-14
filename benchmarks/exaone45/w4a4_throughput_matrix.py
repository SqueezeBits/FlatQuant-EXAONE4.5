#!/usr/bin/env python3
"""Controlled, process-isolated BF16/W4A16/W4A4 vLLM matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
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


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _error(phase, error):
    return {"phase": phase, "type": type(error).__name__, "message": str(error)}


def _base_row(args):
    return {
        "backend": args.backend, "input_length": args.input_length,
        "concurrency": args.concurrency, "output_length": args.output_length,
        "prompt_tokens_per_s": None, "requests_per_s": None,
        "ttft_median_s": None, "ttft_p95_s": None,
        "peak_gpu_memory_bytes": None, "completed_requests": 0,
        "errors": [], "selection_evidence": {},
    }


def _prompt_ids(tokenizer, length):
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    if not ids:
        raise RuntimeError("tokenizer produced no seed tokens")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def _build_llm(args):
    if args.backend == "w4a4":
        import flatquant_vllm_plugin
        flatquant_vllm_plugin.register()
        from flatquant_vllm_plugin.w4a4_ops import selected_w4a4_projections
        selected_w4a4_projections.reset()
    from vllm import LLM
    kwargs = dict(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        enforce_eager=False, trust_remote_code=False,
        max_model_len=args.input_length + args.output_length,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
    )
    if args.backend == "w4a4":
        kwargs["quantization"] = "flatquant_w4a4"
    return LLM(**kwargs)


def run_worker_case(args):
    row = _base_row(args)
    try:
        import torch
        from vllm import SamplingParams
        llm = _build_llm(args)
    except Exception as error:
        row["errors"].append(_error("initialize", error))
        return row
    if args.backend == "w4a4":
        from benchmarks.exaone45.vllm_w4a4 import query_worker_selection
        row["selection_evidence"] = query_worker_selection(llm)
    try:
        prompts = [{"prompt_token_ids": _prompt_ids(llm.get_tokenizer(), args.input_length)}] * args.concurrency
        params = SamplingParams(temperature=0, max_tokens=args.output_length, ignore_eos=True)
        llm.generate(prompts, params, use_tqdm=False)
    except Exception as error:
        row["errors"].append(_error("warmup", error))
        return row
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        ttfts = [float(output.metrics.first_token_latency) for output in outputs
                 if getattr(output, "metrics", None) is not None
                 and output.metrics.first_token_latency is not None]
        row.update({
            "prompt_tokens_per_s": args.input_length * len(outputs) / elapsed,
            "requests_per_s": len(outputs) / elapsed,
            "ttft_median_s": statistics.median(ttfts) if ttfts else None,
            "ttft_p95_s": _percentile(ttfts, .95) if ttfts else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "completed_requests": len(outputs),
        })
    except Exception as error:
        row["errors"].append(_error("measurement", error))
    return row


def _run_case_subprocess(script, backend, model, input_length, concurrency, args):
    with tempfile.TemporaryDirectory(prefix="flatquant-matrix-") as directory:
        result_path = Path(directory) / "row.json"
        command = [
            sys.executable, str(script), "--worker", "--backend", backend,
            "--model", str(model), "--input-length", str(input_length),
            "--concurrency", str(concurrency), "--output-length", str(args.output_length),
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--worker-json", str(result_path),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result_path.exists():
            return json.loads(result_path.read_text())
        row = {
            "backend": backend, "input_length": input_length, "concurrency": concurrency,
            "output_length": args.output_length, "prompt_tokens_per_s": None,
            "requests_per_s": None, "ttft_median_s": None, "ttft_p95_s": None,
            "peak_gpu_memory_bytes": None, "completed_requests": 0,
            "selection_evidence": {},
            "errors": [{"phase": "subprocess", "type": "WorkerExit",
                        "message": f"exit={result.returncode}; stderr={result.stderr[-2000:]}"}],
        }
        return row


def assess_rows(rows, expected_rows):
    reasons = []
    if len(rows) != expected_rows:
        reasons.append(f"expected {expected_rows} rows, received {len(rows)}")
    for row in rows:
        label = f"{row['backend']} input={row.get('input_length')} concurrency={row['concurrency']}"
        if row.get("errors"):
            reasons.append(f"{label} {row['errors'][0]['phase']} failed")
        if row.get("completed_requests") != row["concurrency"]:
            reasons.append(f"{label} completed {row.get('completed_requests')} requests")
        if row["backend"] == "w4a4":
            evidence = row.get("selection_evidence", {})
            if evidence.get("w4a4_projection_count", 0) <= 0:
                reasons.append(f"{label} has no W4A4 model-construction selection evidence")
            for name in ("w4a16_fallback", "bf16_fallback"):
                if evidence.get(name, 0):
                    reasons.append(f"{label} selected {name}")
    return not reasons, reasons


def _format_metric(value, digits):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(payload):
    lines = [f"# EXAONE throughput matrix ({payload['scope']})", "",
             f"Status: `{payload['status']}`", "",
             "| backend | input | concurrency | prompt tokens/s | requests/s | TTFT median | TTFT p95 | peak bytes | completed | errors | selection evidence |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in payload["rows"]:
        errors = "; ".join(f"{x['phase']}:{x['type']}:{x['message']}" for x in row["errors"]) or "none"
        selection = json.dumps(row.get("selection_evidence", {}), sort_keys=True)
        lines.append(
            f"| {row['backend']} | {row['input_length']} | {row['concurrency']} | "
            f"{_format_metric(row['prompt_tokens_per_s'], 3)} | {_format_metric(row['requests_per_s'], 4)} | "
            f"{row['ttft_median_s']} | {row['ttft_p95_s']} | {row['peak_gpu_memory_bytes']} | "
            f"{row['completed_requests']} | {errors} | {selection} |"
        )
    return "\n".join(lines) + "\n"


def _worker_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("bf16", "w4a16", "w4a4"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--worker-json", type=Path, required=True)
    return parser


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        args = _worker_parser().parse_args(argv)
        row = run_worker_case(args)
        args.worker_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        return 0 if not row["errors"] else 3
    args = build_parser().parse_args(argv)
    try:
        paths = validate_model_paths(args.bf16_model, args.w4a16_model, args.w4a4_model)
        if os.environ.get("FLATQUANT_W4A4_STRICT") != "1":
            raise RuntimeError("set FLATQUANT_W4A4_STRICT=1 for the throughput matrix")
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    config = MatrixConfig(tuple(args.input_lengths), tuple(args.concurrencies), args.output_length)
    rows = []
    script = Path(__file__).resolve()
    for backend in ("bf16", "w4a16", "w4a4"):
        for input_length, concurrency in config.cases():
            rows.append(_run_case_subprocess(
                script, backend, paths[backend], input_length, concurrency, args
            ))
    ok, failure_reasons = assess_rows(rows, expected_rows=3 * len(tuple(config.cases())))
    payload = {
        "status": "verified" if ok else "failed", "scope": "real_33b_matrix",
        "environment": _metadata(), "configuration": vars(config),
        "model_paths": {k: str(v) for k, v in paths.items()},
        "failure_reasons": failure_reasons, "rows": rows,
        "process_isolation": "fresh subprocess per backend/input-length/concurrency row",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = args.markdown or args.json.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(payload))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
