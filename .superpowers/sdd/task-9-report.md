# Task 9 report: CUDA Graph and throughput validation harness

## Outcome

Implemented and verified CUDA Graph compatibility with the real tiny
conditional W4A4 fixture. Implemented the controlled 33B throughput matrix
runner, but did not run or fabricate the BF16/W4A16/W4A4 matrix because no real
33B W4A4 artifact exists.

## Verified evidence

- Hardware: NVIDIA A100-SXM4-80GB.
- Software: vLLM 0.24.0, PyTorch 2.11.0+cu130.
- vLLM compiled the dynamic token range `(1, 8192)`.
- vLLM captured 51 mixed prefill/decode piecewise graphs and 35 full decode
  graphs.
- Equal-shape prompt replay with changed token values produced changed logits.
- Allocated CUDA memory was identical before and after measured replay.
- The W4A4 C++ ops now use symbolic Python fake implementations; transform and
  dispatch marker custom ops also expose fake/Meta implementations.
- Model construction selected exactly four unique fused W4A4 prefixes. This is
  backend-selection evidence, not replay-call cardinality.

## Capture issues found and fixed

1. A Python lock in `DispatchCounters.increment` was inside Dynamo fullgraph.
   Graph mode no longer performs invocation accounting. A unique-prefix
   registry records truthful W4A4 selections during model construction.
2. C++ Meta implementations specialized vLLM's dynamic M dimension. They were
   replaced with shared Python `register_fake` functions that retain symbolic
   shapes and preserve the former validation/rejection contract.
3. Python row-threshold comparison specialized symbolic M. Compiled graph mode
   now explicitly requires `FLATQUANT_W4A4_MIN_ROWS=1` and the W4A4-only
   artifact; eager mode retains the diagnostic policy path.
4. Extension schemas must exist before fake registration. `deploy._CUDA` now
   loads before `deploy.nn`.

## Throughput harness

`benchmarks/exaone45/w4a4_throughput_matrix.py` validates all local model paths
and the W4A4 manifest before engine initialization or output creation. For each
controlled backend/length/concurrency row it records prompt tokens/s, aggregate
requests/s, TTFT median and p95, peak allocated GPU memory, completed requests,
structured errors, and selection evidence, then emits JSON and Markdown. Every
backend/shape row runs in a fresh subprocess. Any failed/incomplete row or
missing/invalid W4A4 selection evidence makes the overall status `failed` and
the exit nonzero while preserving diagnostic JSON.

Observed missing-artifact behavior:

```text
exit=2
error: real W4A4 local model directory does not exist: ...; the BF16/W4A16/W4A4 matrix was not run
no_result_written=true
```

## Verification

Focused CUDA Graph test:

```text
1 passed, 9 deselected, 18 warnings in 34.78s
```

Available regression suite:

```text
126 passed, 18 warnings in 28.53s
```

The brief named `tests/test_w4a4_transform_quant.py`, but that file does not
exist at head `8aa666e`; all extant W4A4, exporter, plugin, logits, Triton,
dispatch benchmark, and throughput harness tests were run.

The requested `/workspace/.venvs/flatquant-vllm` environment also does not
exist. Verification used `/workspace/.venv`. The first editable install attempt
was blocked because fast-hadamard-transform hardcodes `compute_70`, unsupported
by CUDA 13. The FlatQuant extension was rebuilt successfully with the existing
FHT build exposed on `PYTHONPATH`.

## Explicitly blocked / not claimed

- No real EXAONE-4.5-33B W4A4 checkpoint was available.
- No BF16/W4A16/W4A4 33B throughput matrix was run.
- No 33B throughput improvement, variance, OOM, accuracy, or serving success
  criterion is claimed.

## Self-review

- Checked that graph evidence is labeled as tiny-fixture evidence everywhere.
- Checked that matrix output cannot be written after missing path validation.
- Checked counter semantics are not represented as replay call counts.
- Checked documentation includes the exact future matrix and serving commands.
- Checked the diff for unrelated changes; generated build products and existing
  untracked review/progress files are not included in the commit.

## Review fixes

- Removed the false-mutation dispatch marker custom op.
- Added full fake/eager CUDA validation parity for contiguity, scale dtype and
  shape, and optional bias dtype/shape/contiguity.
- Strengthened graph replay to prompt lengths 2 and 3, decode lengths 1 and 2,
  and four alternating replays per case with per-replay allocator checks.
- Isolated every throughput row in a fresh process with JSON IPC and structured
  initialization, warm-up, measurement, and worker-exit failures.
- Changed the documented BF16 argument to an explicitly local model path.

Review verification of diagnostic CLI behavior:

```text
missing artifact: exit=2, no_result_written=true
failed worker rows: exit=3, status=failed, rows=3, all errors structured
```
