"""Offline vLLM generation and latency runner for the EXAONE-4.5 AWQ model."""

import argparse
import contextlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path


DEFAULT_AWQ_PATH = (
    "/workspace/.hf_home/hub/"
    "models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/"
    "snapshots/d73d64aa670777f94f101916ea0803e033ba9b59"
)


def _engine_kwargs(args):
    return dict(
        model=args.model_path,
        tokenizer=args.tokenizer or args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=False,
    )


def _build_engine(args):
    from vllm import LLM

    return LLM(**_engine_kwargs(args))


def _prompt_text(tokenizer, prompt, target_tokens):
    if target_tokens is None:
        return prompt
    seed = tokenizer.encode(prompt, add_special_tokens=False)
    if not seed:
        raise ValueError("Prompt must tokenize to at least one token.")
    token_ids = (seed * ((target_tokens + len(seed) - 1) // len(seed)))[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def run_generate(args):
    from vllm import SamplingParams

    llm = _build_engine(args)
    prompt = _prompt_text(llm.get_tokenizer(), args.prompt, args.prefill_tokens)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate([prompt], params, use_tqdm=False)
    print(outputs[0].outputs[0].text)


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _distribution(values, prefix):
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_mean": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_p90": _percentile(values, 0.90),
        f"{prefix}_max": max(values),
    }


def summarize_latency(samples):
    """Summarize completed generation observations without importing vLLM."""
    if not samples:
        raise ValueError("At least one latency sample is required.")

    elapsed = [sample["elapsed_s"] for sample in samples]
    first_token = [
        sample["first_token_s"]
        for sample in samples
        if sample["first_token_s"] is not None
    ]
    tpot_ms = [
        1000.0
        * (sample["elapsed_s"] - sample["first_token_s"])
        / (sample.get("output_tokens_per_request", sample["output_tokens"]) - 1)
        for sample in samples
        if sample["first_token_s"] is not None
        and sample.get("output_tokens_per_request", sample["output_tokens"]) > 1
    ]
    input_throughput = [
        sample["input_tokens"] / sample["elapsed_s"] for sample in samples
    ]
    output_throughput = [
        sample["output_tokens"] / sample["elapsed_s"] for sample in samples
    ]

    result = {
        "ttft_source": (
            "vllm_request_metrics" if len(first_token) == len(samples) else "unavailable"
        )
    }
    result.update(_distribution(first_token, "ttft_s"))
    result.update(_distribution(tpot_ms, "tpot_ms"))
    result.update(_distribution(elapsed, "e2e_s"))
    result.update(_distribution(input_throughput, "input_tokens_per_s"))
    result.update(_distribution(output_throughput, "output_tokens_per_s"))
    return result


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_metadata():
    import torch

    metadata = {
        "python_packages": {
            name: _package_version(name)
            for name in ("torch", "triton", "vllm", "flatquant-vllm-plugin")
        },
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        }
    return metadata


def _first_token_latency(outputs):
    values = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        value = getattr(metrics, "first_token_latency", None)
        if value is None:
            return None
        values.append(value)
    return statistics.median(values) if values else None


def _nvtx_measurement_range(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        return contextlib.nullcontext()
    return torch_module.cuda.nvtx.range("vllm_measurement")


def run_latency(args):
    from vllm import SamplingParams

    llm = _build_engine(args)
    prompt = _prompt_text(llm.get_tokenizer(), args.prompt, args.prefill_tokens)
    prompts = [prompt] * args.batch_size
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    for _ in range(args.warmup_steps):
        llm.generate(prompts, params, use_tqdm=False)

    input_tokens = sum(len(token_ids) for token_ids in llm.get_tokenizer()(prompts).input_ids)
    samples = []
    with _nvtx_measurement_range():
        for _ in range(args.num_repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            generated_per_request = [
                len(output.outputs[0].token_ids) for output in outputs
            ]
            generated = sum(generated_per_request)
            samples.append(
                {
                    "first_token_s": _first_token_latency(outputs),
                    "elapsed_s": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": generated,
                    "output_tokens_per_request": max(generated_per_request, default=0),
                }
            )

    result = {
        "backend": "vllm",
        "model": args.model_path,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prefill_tokens": args.prefill_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_steps": args.warmup_steps,
        "num_repeats": args.num_repeats,
        "enforce_eager": args.enforce_eager,
        "environment": _environment_metadata(),
        "samples": samples,
    }
    result.update(summarize_latency(samples))
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "latency"))
    parser.add_argument("--model_path", default=DEFAULT_AWQ_PATH)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16"))
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--prompt", default="Explain quantization in one paragraph.")
    parser.add_argument("--prefill_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--output_json", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        run_latency(args)


if __name__ == "__main__":
    main()
