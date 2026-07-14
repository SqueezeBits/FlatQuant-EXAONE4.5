# Task 8 report: decode dispatch policy and observability

## Result

- Added `DispatchPolicy.from_env()` with safe defaults: `min_w4a4_rows=1` and
  strict mode disabled unless `FLATQUANT_W4A4_STRICT=1`.
- Dispatch uses flattened M and layer prefix. Strict rejection reports prefix,
  M, and the selected fallback.
- Counters expose a stable RPC-friendly schema: `w4a4`, `w4a16_fallback`, and
  `bf16_fallback`, including after reset.
- Export manifests declare only `representations: ["w4a4"]`. No W4A16/BF16
  tensors are duplicated. Because this scope has no matching fallback exporter
  and load path, fallback declarations and requested fallback dispatch reject
  explicitly instead of fabricating a representation.
- Preserved direct `apply_w4a4()` compatibility and the tiny exact vLLM
  dispatch test.

## TDD evidence

The first dispatch test run under the correct venv/PYTHONPATH failed at import:
`cannot import name 'DispatchPolicy'`. After implementation:

```text
pytest -q tests/test_vllm_w4a4_plugin.py -k dispatch
8 passed, 16 deselected
```

## A100 crossover

GPU: NVIDIA A100-SXM4-80GB. Protocol: 10 warmups and 30 CUDA-event samples per
representation/shape/M. M was `1,2,4,8,16,32,64,128`. Every path included the
same learned FlatQuant transform; native W4A4 additionally included activation
quantize/pack, W4A16 used Marlin, and BF16 used BF16 linear. Shapes were
the actual fused EXAONE 4.5 32B QKV `(7168,5120)`, O `(5120,5120)`, gate/up
`(54784,5120)`, and down `(5120,27392)`.

Down projection selected W4A16 through M=64; M=128 was the first measured point
where W4A4 won across every target projection. Since 128 is the final requested
sample, this run does not establish stability beyond the crossover. It is
documented but deliberately not encoded: the W4A4-only artifact remains at the
safe threshold 1 because it has no legal fallback. Raw results are in
`.superpowers/sdd/task-8-a100-crossover.json` and the reproducible harness is
`benchmarks/w4a4_dispatch_benchmark.py`.

## Verification

```text
FLATQUANT_W4A4_STRICT=1 pytest -q tests/test_vllm_w4a4_plugin.py tests/test_exaone45_w4a4_logits.py
33 passed

FLATQUANT_W4A4_STRICT=0 pytest -q tests/test_vllm_w4a4_plugin.py -k fallback
1 passed, 23 deselected

pytest -q tests/test_vllm_w4a4_plugin.py tests/test_export_flatquant_w4a4_vllm.py
29 passed
```

Warnings were pre-existing PyTorch/FlashInfer deprecations plus the tiny vLLM
test's process-group shutdown warning; there were no failures.

## Review

Review requested against base `04a8a3f`, focusing on manifest truthfulness,
strict selection, counters, benchmark fairness, and weight duplication.
Review identified the stale worker strict gate, which checked the legacy
`fallback` name; it was corrected with RED/GREEN tests for both new fallback
counters. Its request for a working fallback was intentionally not adopted:
the parent requirement explicitly mandates policy rejection when no truthful
exporter/runtime dual representation exists. The review also prompted inclusion
of the learned transform in every microbenchmark path and a clear limitation
that M=128 is the final sample, not evidence of persistence beyond it.

## Review follow-up

- `validate_manifest` and runtime config now require
  `representations == ["w4a4"]`; missing, empty, duplicate, W4A16, and BF16
  declarations are rejected.
- `FLATQUANT_W4A4_STRICT` now accepts only literal `0` or `1`.
  `FLATQUANT_W4A4_MIN_ROWS` gives a named error for non-integers and rejects
  zero/negative values.
- The benchmark now accepts `--model` and `--transform-checkpoint`, resolves a
  Hugging Face cache root through `refs/main`, reads the safetensors index, and
  requires exact exported layer-0 tensor names for QKV, O, gate/up, and down.
  Missing tensors fail closed; there is no identity fallback.
- Raw JSON was regenerated from
  `/workspace/.hf_home/hub/models--Hyun9junn--EXAONE-4.5-33B-FlatQuant-W4A16/snapshots/61571ec565670a5e8304b13473ba33793e61204d`.
  With the real learned transform applied equally to every path, QKV, O, and
  gate/up selected W4A4 at every sampled M. Down selected W4A16 at M=1 through
  64 and W4A4 at M=128. Therefore M=128 remains the first sampled all-projection
  W4A4 win, without a stability claim beyond the requested range.

Follow-up verification:

```text
pytest -q tests/test_w4a4_format.py tests/test_w4a4_dispatch_benchmark.py tests/test_export_flatquant_w4a4_vllm.py
21 passed

FLATQUANT_W4A4_STRICT=1 pytest -q tests/test_vllm_w4a4_plugin.py tests/test_exaone45_w4a4_logits.py
40 passed

FLATQUANT_W4A4_STRICT=0 pytest -q tests/test_vllm_w4a4_plugin.py -k fallback
1 passed, 30 deselected
```
