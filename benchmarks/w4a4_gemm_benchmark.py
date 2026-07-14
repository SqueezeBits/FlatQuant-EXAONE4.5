"""Benchmark the fused SM80 W4A4 GEMM on real EXAONE projection shapes."""
import argparse
import json
from pathlib import Path

import torch

from flatquant_w4a4.packing import pack_signed_i4
import deploy._CUDA  # noqa: F401


def projection_shapes(config):
    text = config.get("text_config", config)
    hidden = int(text["hidden_size"])
    intermediate = int(text["intermediate_size"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text.get("head_dim", hidden // heads))
    return sorted({
        (hidden, hidden), (kv_heads * head_dim, hidden),
        (intermediate, hidden), (hidden, intermediate),
    })


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--rows", nargs="+", type=int, required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("this benchmark requires an A100/SM80 GPU")
    config = json.loads(Path(args.model_config).read_text())
    records = []
    torch.manual_seed(7)
    for n, k in projection_shapes(config):
        qweight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
        packed_w = pack_signed_i4(qweight).cuda()
        wscale = (torch.rand(n, 1, dtype=torch.float16) * 0.02).cuda()
        for m in args.rows:
            packed_x = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
            xscale = torch.rand(m, 1, device="cuda") * 0.02
            call = lambda: torch.ops.flatquant.w4a4_linear(
                packed_x, packed_w, xscale, wscale, torch.bfloat16)
            for _ in range(args.warmup):
                output = call()
            torch.cuda.synchronize()
            times = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); output = call(); end.record(); end.synchronize()
                times.append(start.elapsed_time(end))
            # Check a deterministic row against an exact integer CPU reference.
            bytes0 = packed_x[0].cpu()
            lo = (bytes0 & 15).to(torch.int8); hi = (bytes0 >> 4).to(torch.int8)
            lo[lo >= 8] -= 16; hi[hi >= 8] -= 16
            qx = torch.stack((lo, hi), dim=1).reshape(k).to(torch.int32)
            ref = (qx @ qweight.to(torch.int32).T).float().cuda()
            ref *= xscale[0, 0] * wscale[:, 0].float()
            max_error = (output[0].float() - ref).abs().max().item()
            median = percentile(times, 0.5)
            candidate_medians = []
            candidate_medians = [None, None, None]
            offset = (m + n + k) % 3
            candidate_order = [(offset + i) % 3 for i in range(3)]
            for candidate in candidate_order:
                for _ in range(args.warmup):
                    torch.ops.flatquant._w4a4_linear_candidate(
                        packed_x, packed_w, xscale, wscale,
                        torch.bfloat16, candidate)
                candidate_times = []
                for _ in range(args.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    torch.ops.flatquant._w4a4_linear_candidate(
                        packed_x, packed_w, xscale, wscale,
                        torch.bfloat16, candidate)
                    end.record(); end.synchronize()
                    candidate_times.append(start.elapsed_time(end))
                candidate_medians[candidate] = percentile(candidate_times, 0.5)
            records.append({
                "M": m, "N": n, "K": k,
                "kernel": torch.ops.flatquant.w4a4_kernel_name(m, n, k),
                "median_ms": median, "p95_ms": percentile(times, 0.95),
                "effective_tops": (2.0 * m * n * k) / (median * 1e9),
                "max_error": max_error,
                "candidate_median_ms": candidate_medians,
                "candidate_timing_order": candidate_order,
                "measured_winner": min(
                    range(3), key=candidate_medians.__getitem__),
            })
    path = Path(args.json); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "results": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
