# Task 6 report: transform + activation packing fusion

## Status

An experimental fused-memory candidate was implemented and tested, but the
required A100 prefill benchmark gate failed. The rejected runtime, its tests,
and its benchmark CLI were removed after review. Production dispatch remains
the faster composed transform + Task 3 CUDA quantizer, with no retained Task 6
runtime complexity.

## RED

`PYTHONPATH=vllm_plugin pytest -q tests/test_w4a4_transform_quant.py` failed
during collection with:

```
ModuleNotFoundError: No module named 'flatquant_vllm_plugin.w4a4_transform_quant'
```

## Experimental result (subsequently removed)

The candidate used a two-launch recomputation boundary:

1. Transform tiles reduce into per-row FP32 maxima with atomics.
2. Transform tiles are recomputed, clipped, rounded with `div.rn` plus
   `float2int_rn`, clamped, and packed low nibble first.

The experiment genuinely removed the `[M, K]` transformed global allocation
and its subsequent read; it was not merely a composition of existing operators.
Its output buffers were `[M, K/2] uint8` and `[M, 1] float32`, matching the
Task 3 operator and `w4a4_linear` ABI. The task brief's FP16 scale statement is
incompatible with the checked-in Task 3 binding, which explicitly requires
FP32 activation scales. The removed candidate covered both left-only and
Kronecker variants.

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

These were candidate-validation results, not tests retained in the final tree.
Strict dispatch counters remained unchanged because production `apply_w4a4`
never retained the experimental path.

## A100 benchmark

Historical command used before removal:

```
PYTHONPATH=.:vllm_plugin python benchmarks/kernel_benchmark.py \
  --kernel w4a4-transform-quant --rows 128 512 1024 4096
```

The temporary harness performed 20 warmups and 100 individually timed iterations and
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

- Only the measured evidence is retained in this report. The rejected runtime,
  candidate-only tests, and benchmark plumbing are absent from the final tree.
- A faster dense-transform fusion would need a different cross-program row
  reduction mechanism or a downstream GEMM interface that consumes tiled
  scales; neither is a small safe change to the Task 3 ABI.
- Benchmark allocation is outside the measured synchronization boundary only
  for persistent inputs; operator output allocation remains part of both paths,
  reflecting runtime use.

## Final post-removal regression

After removing all rejected candidate code and restoring
`benchmarks/kernel_benchmark.py` byte-for-byte to its pre-Task 6 version:

```
PYTHONPATH=.:vllm_plugin pytest -q \
  tests/test_vllm_w4a4_plugin.py tests/test_exaone45_w4a4_logits.py

25 passed, 18 warnings in 15.14s
```

This verifies the composed production path, strict dispatch accounting, and
EXAONE logits/load/generation gates without any candidate runtime dependency.
