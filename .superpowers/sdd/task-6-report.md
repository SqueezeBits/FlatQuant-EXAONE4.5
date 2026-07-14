# Task 6 report: transform + activation packing fusion

## Status

Implemented and correctness-tested a genuine fused-memory candidate, but did
not enable it in `apply_w4a4`: the required A100 prefill benchmark gate failed.
Production dispatch therefore remains the faster composed transform + Task 3
CUDA quantizer.

## RED

`PYTHONPATH=vllm_plugin pytest -q tests/test_w4a4_transform_quant.py` failed
during collection with:

```
ModuleNotFoundError: No module named 'flatquant_vllm_plugin.w4a4_transform_quant'
```

## GREEN

`w4a4_transform_quant.py` implements a two-launch recomputation boundary:

1. Transform tiles reduce into per-row FP32 maxima with atomics.
2. Transform tiles are recomputed, clipped, rounded with `div.rn` plus
   `float2int_rn`, clamped, and packed low nibble first.

This truly removes the `[M, K]` transformed global allocation and its subsequent
read. It does not label a composition of the existing operators as fused.
Fixed output buffers are `[M, K/2] uint8` and `[M, 1] float32`, matching the
Task 3 operator and `w4a4_linear` ABI. The task brief's FP16 scale statement is
incompatible with the checked-in Task 3 binding, which explicitly requires
FP32 activation scales. Fake/meta custom-op implementations cover both
left-only and Kronecker variants.

Exactness covers M=1,16,128,1024, clip=0.625/boundary values, non-contiguous API
input, and actual EXAONE factor families: left-only 40 (K=6400), 64x80, and
128x214. Packed bytes match exactly; scales use rtol=1e-3, atol=1e-5.

Verification:

```
31 passed, 14 warnings in 7.90s
# tests/test_w4a4_transform_quant.py + tests/test_vllm_w4a4_plugin.py

7 passed, 18 warnings in 14.45s
# tests/test_exaone45_w4a4_logits.py
```

Strict dispatch counters are unchanged because production `apply_w4a4` remains
on its existing path.

## A100 benchmark

Command:

```
PYTHONPATH=.:vllm_plugin python benchmarks/kernel_benchmark.py \
  --kernel w4a4-transform-quant --rows 128 512 1024 4096
```

The harness performs 20 warmups and 100 individually timed iterations and
checks exact packed output before timing.

| rows | composed median us | composed p95 us | candidate median us | candidate p95 us |
|---:|---:|---:|---:|---:|
| 128 | 129.024 | 142.336 | 125.952 | 137.216 |
| 512 | 128.000 | 140.288 | 154.624 | 163.840 |
| 1024 | 144.384 | 155.648 | 207.872 | 222.208 |
| 4096 | 351.232 | 359.424 | 537.600 | 543.744 |

The candidate wins 2.4% at 128 rows, then loses 20.8%, 44.0%, and 53.1% at the
representative prefill buckets. Recomputing dense learned transforms costs more
than the eliminated memory traffic, so the production switch was reverted.

## Self-review / concerns

- The candidate is deliberately retained as measured evidence and a reusable
  exactness baseline, not enabled or described as a production speedup.
- A faster dense-transform fusion would need a different cross-program row
  reduction mechanism or a downstream GEMM interface that consumes tiled
  scales; neither is a small safe change to the Task 3 ABI.
- Benchmark allocation is outside the measured synchronization boundary only
  for persistent inputs; operator output allocation remains part of both paths,
  reflecting runtime use.
