"""Sweep bounded Triton configurations for FlatQuant Kronecker transforms."""

import argparse
import json
from pathlib import Path

import torch
import triton

from flatquant_vllm_plugin.triton_transform_v2 import (
    KronConfig,
    _fused_kron_kernel,
)


SHAPES = ((64, 80), (128, 214))
TOKENS = (1, 2, 4, 8, 16, 64, 256, 512, 2048)


def candidate_configs():
    return [
        KronConfig(block_n=block_n, num_warps=num_warps)
        for block_n in (16, 32, 64)
        for num_warps in (4, 8)
    ]


def _reference(x, left, right):
    shape = x.shape
    value = x.reshape(-1, left.shape[0], right.shape[0]).float()
    value = torch.matmul(value, right.float())
    value = torch.matmul(left.T.float(), value)
    return value.reshape(shape)


def _launch(x, left, right, output, config):
    tokens = x.numel() // x.shape[-1]
    left_size = left.shape[0]
    right_size = right.shape[0]
    l_pad = triton.next_power_of_2(left_size)
    r_pad = triton.next_power_of_2(right_size)
    block_n = min(config.block_n, r_pad)
    value = x.reshape(tokens * left_size, right_size)
    _fused_kron_kernel[(tokens, triton.cdiv(right_size, block_n))](
        value,
        left,
        right,
        output,
        left_size=left_size,
        right_size=right_size,
        L_PAD=l_pad,
        R_PAD=r_pad,
        BLOCK_N=block_n,
        num_warps=config.num_warps,
    )


def benchmark_case(tokens, left_size, right_size, config, warmup_ms, rep_ms):
    torch.manual_seed(0)
    x = torch.randn(
        tokens,
        left_size * right_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    left = torch.randn(left_size, left_size, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(
        right_size, right_size, device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty_like(x)
    reference = _reference(x, left, right)

    try:
        _launch(x, left, right, output, config)
        torch.cuda.synchronize()
        scale = reference.abs().max().clamp_min(1e-6)
        relative_max_error = (
            (output.float() - reference).abs().max() / scale
        ).item()
        if relative_max_error > 5e-3:
            raise ValueError(
                f"relative max error {relative_max_error:.6f} exceeds 5e-3"
            )
        quantiles = triton.testing.do_bench(
            lambda: _launch(x, left, right, output, config),
            warmup=warmup_ms,
            rep=rep_ms,
            quantiles=(0.2, 0.5, 0.8),
        )
        return {
            "status": "ok",
            "relative_max_error": relative_max_error,
            "latency_us_p20": quantiles[0] * 1000.0,
            "latency_us_median": quantiles[1] * 1000.0,
            "latency_us_p80": quantiles[2] * 1000.0,
        }
    except Exception as error:
        torch.cuda.synchronize()
        return {"status": "rejected", "error": f"{type(error).__name__}: {error}"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    rows = []
    for left_size, right_size in SHAPES:
        for tokens in args.tokens:
            for config in candidate_configs():
                result = benchmark_case(
                    tokens,
                    left_size,
                    right_size,
                    config,
                    args.warmup_ms,
                    args.rep_ms,
                )
                row = {
                    "tokens": tokens,
                    "left_size": left_size,
                    "right_size": right_size,
                    "block_n": config.block_n,
                    "num_warps": config.num_warps,
                    **result,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "rows": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
