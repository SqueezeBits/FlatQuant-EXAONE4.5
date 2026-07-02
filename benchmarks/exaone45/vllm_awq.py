"""Offline vLLM generation and latency runner for the EXAONE-4.5 AWQ model."""

import argparse
import json
import statistics
import time
from pathlib import Path


DEFAULT_AWQ_PATH = (
    "/workspace/.hf_home/hub/"
    "models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/"
    "snapshots/d73d64aa670777f94f101916ea0803e033ba9b59"
)


def _build_engine(args):
    from vllm import LLM

    return LLM(
        model=args.model_path,
        tokenizer=args.tokenizer or args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
    )


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


def run_latency(args):
    from vllm import SamplingParams

    llm = _build_engine(args)
    prompt = _prompt_text(llm.get_tokenizer(), args.prompt, args.prefill_tokens)
    prompts = [prompt] * args.batch_size
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    for _ in range(args.warmup_steps):
        llm.generate(prompts, params, use_tqdm=False)

    elapsed = []
    generated = []
    for _ in range(args.num_repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed.append(time.perf_counter() - start)
        generated.append(sum(len(output.outputs[0].token_ids) for output in outputs))

    result = {
        "backend": "vllm",
        "model": args.model_path,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prefill_tokens": args.prefill_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_steps": args.warmup_steps,
        "num_repeats": args.num_repeats,
        "latency_s_mean": statistics.mean(elapsed),
        "latency_s_median": statistics.median(elapsed),
        "latency_s_p90": _percentile(elapsed, 0.90),
        "output_tokens_per_s_mean": statistics.mean(
            tokens / duration for tokens, duration in zip(generated, elapsed)
        ),
    }
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
