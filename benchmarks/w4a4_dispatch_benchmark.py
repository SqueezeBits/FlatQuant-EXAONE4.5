"""Compare decode GEMM representations on the four EXAONE 4.5 projections."""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch

import deploy._CUDA  # noqa: F401
from deploy.nn import LinearW4A16Marlin
from flatquant_w4a4.packing import pack_signed_i4
from flatquant_vllm_plugin.transform import apply_transform, decompose_dim


ROWS = (1, 2, 4, 8, 16, 32, 64, 128)


def elapsed(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100/SM80 is required")
    config = json.loads(args.config.read_text()).get("text_config")
    h, i = config["hidden_size"], config["intermediate_size"]
    head_dim = config.get("head_dim") or h // config["num_attention_heads"]
    kv = config["num_key_value_heads"] * head_dim
    shapes = {
        "qkv_proj": (h + 2 * kv, h),
        "o_proj": (h, h),
        "gate_up_proj": (2 * i, h),
        "down_proj": (h, i),
    }
    results = []
    torch.manual_seed(8)
    for projection, (n, k) in shapes.items():
        qweight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
        packed = pack_signed_i4(qweight)
        scale = torch.rand(n, 1, dtype=torch.float16) * 0.02
        marlin = LinearW4A16Marlin(k, n, output_dtype=torch.bfloat16).cuda()
        marlin.load_packed_weight(packed, scale, "cuda")
        packed, scale = packed.cuda(), scale.cuda()
        bf16_weight = (
            qweight.cuda().to(torch.bfloat16) * scale.to(torch.bfloat16)
        ).contiguous()
        if projection == "o_proj":
            transform_layer = SimpleNamespace(
                flatquant_left=torch.eye(40, device="cuda", dtype=torch.bfloat16)
            )
        else:
            left, right = decompose_dim(k)
            transform_layer = SimpleNamespace(
                flatquant_left=torch.eye(
                    left, device="cuda", dtype=torch.bfloat16
                ),
                flatquant_right=torch.eye(
                    right, device="cuda", dtype=torch.bfloat16
                ),
            )
        del qweight
        for m in ROWS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            clip = torch.tensor([3.0], device="cuda", dtype=torch.float16)

            def w4a4():
                transformed = apply_transform(transform_layer, x)
                qx, xs = torch.ops.flatquant.quantize_pack_i4(transformed, clip)
                return torch.ops.flatquant.w4a4_linear(
                    qx, packed, xs, scale, torch.bfloat16
                )

            calls = {
                "w4a4": w4a4,
                "w4a16": lambda: marlin(apply_transform(transform_layer, x)),
                "bf16": lambda: torch.nn.functional.linear(
                    apply_transform(transform_layer, x), bf16_weight
                ),
            }
            times = {
                name: elapsed(call, args.warmup, args.iterations)
                for name, call in calls.items()
            }
            results.append({"projection": projection, "M": m, "N": n, "K": k,
                            "median_ms": times})
        del marlin, packed, scale, bf16_weight
        torch.cuda.empty_cache()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(), "warmup": args.warmup,
        "iterations": args.iterations, "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
