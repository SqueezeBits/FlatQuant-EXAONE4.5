# FlatQuant W4A16 vLLM Latency Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce EXAONE-4.5 FlatQuant W4A16 vLLM TPOT and prefill latency through measured, correctness-preserving optimizations.

**Architecture:** Establish a machine-readable end-to-end baseline first, then optimize the existing custom-op Triton transform without changing checkpoint semantics. Runtime changes remain behind the existing FlatQuant quantization method; offline selective-transform and Marlin-fusion work are isolated experiments that cannot silently alter the default path.

**Tech Stack:** Python 3.12, PyTorch 2.11, Triton 3.6, vLLM 0.24, CUDA 13.0, Nsight Systems, Nsight Compute, unittest/pytest.

## Global Constraints

- Primary workload: batch 1, 64 input tokens, 128 output tokens.
- Secondary workload: batch 4, 512 input tokens, 16 output tokens.
- Preserve the existing FlatQuant checkpoint path and default behavior.
- Do not disable an online transform for weights exported in transformed coordinates.
- Production comparisons run without `--enforce-eager` and must record CUDA Graph state.
- Transform relative maximum error must remain at or below `5e-3` against the FP32 reference.
- Make one hypothesis-driven runtime change at a time and retain it only after isolated A/B evidence.
- Do not install or modify the NVIDIA driver.

---

## File Map

- `benchmarks/exaone45/vllm_awq.py`: end-to-end fixed-workload runner and JSON metrics.
- `benchmarks/transform_kernel_benchmark.py`: standalone transform sweep and JSON results.
- `benchmarks/exaone45/profile_vllm.sh`: repeatable Nsight Systems capture wrapper.
- `tests/test_vllm_awq_benchmark.py`: pure unit tests for metric calculation and result schema.
- `tests/test_triton_transform.py`: CUDA correctness and configuration-selection tests.
- `vllm_plugin/flatquant_vllm_plugin/triton_transform_v2.py`: transform layout handling and tuned Triton dispatch.
- `vllm_plugin/flatquant_vllm_plugin/config.py`: one-time transform-factor layout normalization.
- `tools/export_flatquant_vllm.py`: optional projection/layer selection during a new offline export.
- `tests/test_export_flatquant_vllm.py`: selection parsing and export-safety tests.
- `experiments/marlin_fusion/README.md`: measured small-M fusion feasibility decision.
- `outputs/benchmark_results/`: untracked runtime JSON and profiler artifacts.

### Task 1: Reproducible End-to-End Metrics

**Files:**
- Modify: `benchmarks/exaone45/vllm_awq.py`
- Create: `tests/test_vllm_awq_benchmark.py`

**Interfaces:**
- Produces: `summarize_latency(samples, input_tokens, output_tokens) -> dict`
- Produces: JSON fields `ttft_s_*`, `tpot_ms_*`, `e2e_s_*`, `input_tokens_per_s_median`, `output_tokens_per_s_median`, and environment metadata.

- [ ] **Step 1: Write failing pure metric tests**

Create tests that pass synthetic per-request records containing `first_token_s`,
`elapsed_s`, `input_tokens`, and `output_tokens`. Assert exact median TTFT, TPOT
computed as `(elapsed_s - first_token_s) / max(output_tokens - 1, 1)`, end-to-end
latency, and throughput. Add a zero/one-output-token case that never divides by
zero.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/workspace/.venv/bin/python -m pytest tests/test_vllm_awq_benchmark.py -q
```

Expected: failure because `summarize_latency` and the expanded schema do not
exist.

- [ ] **Step 3: Implement metrics and metadata**

Refactor metric aggregation into pure functions. Collect version metadata with
`importlib.metadata`, CUDA device properties with PyTorch, command arguments,
and `enforce_eager`. Use vLLM request metrics for first-token timing when exposed
by vLLM 0.24; otherwise report `ttft_s_*: null` and mark
`ttft_source: "unavailable"` rather than estimating TTFT from total latency.
Record min, median, mean, p90, and max for repeated observations.

- [ ] **Step 4: Verify GREEN and existing CLI parsing**

Run:

```bash
/workspace/.venv/bin/python -m pytest tests/test_vllm_awq_benchmark.py -q
/workspace/.venv/bin/python benchmarks/exaone45/vllm_awq.py --help
```

Expected: all metric tests pass and help exits zero.

- [ ] **Step 5: Commit Task 1**

```bash
git add benchmarks/exaone45/vllm_awq.py tests/test_vllm_awq_benchmark.py
git commit -m "bench: record vLLM TTFT and TPOT metrics"
```

### Task 2: Baseline, CUDA Graph Audit, and Profiles

**Files:**
- Create: `benchmarks/exaone45/profile_vllm.sh`
- Create: `outputs/benchmark_results/baseline/` at runtime only

**Interfaces:**
- Consumes: Task 1 JSON schema.
- Produces: FlatQuant eager/non-eager baseline JSON, AWQ baseline when available, Nsight `.nsys-rep`, and a text summary of graph capture and kernel time.

- [ ] **Step 1: Verify model and plugin inputs without loading weights**

Run:

```bash
FQ=/workspace/.hf_home/hub/models--Hyun9junn--EXAONE-4.5-33B-FlatQuant-W4A16/snapshots/61571ec565670a5e8304b13473ba33793e61204d
/workspace/.venv/bin/python -c 'import json,sys; c=json.load(open(sys.argv[1]+"/config.json")); print(c["quantization_config"]["quant_method"])' "$FQ"
/workspace/.venv/bin/python -c 'import vllm,flatquant_vllm_plugin; print(vllm.__version__, flatquant_vllm_plugin.__file__)'
```

Expected: quantization method `flatquant`, vLLM `0.24.0`, and the plugin path.

- [ ] **Step 2: Run primary FlatQuant eager diagnostic**

Run Task 1 runner with `--enforce_eager --prefill_tokens 64
--max_new_tokens 128 --batch_size 1 --warmup_steps 2 --num_repeats 7`, saving
JSON under `outputs/benchmark_results/baseline/flatquant-primary-eager.json`.

- [ ] **Step 3: Run primary and secondary non-eager baselines**

Run the same primary command without `--enforce_eager`, then run batch 4,
prefill 512, output 16. Capture full logs with `tee`; verify logs contain CUDA
Graph capture and do not contain repeated recompilation or graph-break messages.

- [ ] **Step 4: Add and validate the Nsight wrapper**

Create a strict Bash wrapper that accepts model path, workload label, and runner
arguments, invokes `nsys profile --trace=cuda,nvtx,osrt --sample=none
--cpuctxsw=none`, and writes into `outputs/benchmark_results/profiles/`. Validate
with `bash -n benchmarks/exaone45/profile_vllm.sh` before running it.

- [ ] **Step 5: Profile FlatQuant non-eager decode**

Capture one warm run with three measurements. Run `nsys stats` reports for CUDA
GPU kernel summary, CUDA API summary, and NVTX summary. Record transform kernel,
Marlin, attention, allocation/copy calls, and graph-launch totals in
`outputs/benchmark_results/baseline/analysis.md`.

- [ ] **Step 6: Acquire or locate AWQ only if absent, then run identical baselines**

Use the existing `DEFAULT_AWQ_PATH` if present. If absent, download
`LGAI-EXAONE/EXAONE-4.5-33B-AWQ` into the configured HF cache, then run the same
primary and secondary non-eager commands in fresh processes. Do not compare an
eager AWQ run with a graphed FlatQuant run.

- [ ] **Step 7: Commit only the reusable profiler wrapper**

```bash
git add benchmarks/exaone45/profile_vllm.sh
git commit -m "bench: add repeatable vLLM Nsight profile"
```

### Task 3: Eliminate Proven Layout Copies

**Files:**
- Modify: `tests/test_triton_transform.py`
- Modify: `vllm_plugin/flatquant_vllm_plugin/triton_transform_v2.py`
- Modify: `vllm_plugin/flatquant_vllm_plugin/config.py`

**Interfaces:**
- Produces: `_prepare_transform_input(x) -> Tensor` with identity behavior for contiguous input and a safe contiguous fallback.
- Preserves: `kron_transform(x, left, right)` and `left_transform(x, left)` signatures.

- [ ] **Step 1: Write failing layout tests**

Add CPU-callable helper tests asserting that contiguous input returns the same
storage pointer, non-contiguous last-dimension layouts are copied to contiguous
storage, and normalized transform factors remain contiguous after weight-load
processing. Add CUDA correctness coverage for both paths.

- [ ] **Step 2: Run tests and verify RED**

```bash
/workspace/.venv/bin/python -m pytest tests/test_triton_transform.py -q
```

Expected: failure because `_prepare_transform_input` and factor normalization do
not exist.

- [ ] **Step 3: Implement the minimum layout change**

Introduce `_prepare_transform_input` and replace unconditional expression-level
layout calls with explicit prepared tensors. Normalize factor `.data` once in
`process_weights_after_loading`; retain the fallback for non-contiguous inputs.
Do not change kernel math or block configuration in this task.

- [ ] **Step 4: Verify correctness and A/B performance**

Run the full transform tests, then repeat Task 2 primary and secondary FlatQuant
benchmarks. Use profiler evidence to confirm whether copy/allocation counts
changed. Revert the runtime edit if no copy existed and latency does not improve
beyond run-to-run noise.

- [ ] **Step 5: Commit only if retained**

```bash
git add tests/test_triton_transform.py vllm_plugin/flatquant_vllm_plugin/triton_transform_v2.py vllm_plugin/flatquant_vllm_plugin/config.py
git commit -m "perf: avoid repeated FlatQuant layout preparation"
```

### Task 4: Bounded Shape-Aware Triton Configuration

**Files:**
- Create: `benchmarks/transform_kernel_benchmark.py`
- Modify: `tests/test_triton_transform.py`
- Modify: `vllm_plugin/flatquant_vllm_plugin/triton_transform_v2.py`

**Interfaces:**
- Produces: `select_kron_config(left_size: int, right_size: int, tokens: int) -> KronConfig`.
- Produces: JSON sweep records containing shape, tokens, block size, warps, median microseconds, and reference error.

- [ ] **Step 1: Write failing deterministic selection tests**

Test exact bucket boundaries for tokens `1, 16, 17, 256, 257, 2048`, both
EXAONE factor shapes, and unknown-shape fallback. Assert the selected values are
from the bounded candidate set and stable between calls.

- [ ] **Step 2: Verify RED**

```bash
/workspace/.venv/bin/python -m pytest tests/test_triton_transform.py -q
```

Expected: failure because `KronConfig` and `select_kron_config` do not exist.

- [ ] **Step 3: Add the standalone sweep before changing dispatch**

Implement warm-up and at least 100 timed iterations with CUDA events for token
counts `1,2,4,8,16,64,256,512,2048`, shapes `(64,80)` and `(128,214)`, block
sizes `16,32,64`, and warp counts `4,8`. Reject configurations exceeding the
`5e-3` reference-error gate. Save raw JSON; do not select based on one timing.

- [ ] **Step 4: Run the sweep and inspect top configurations**

```bash
PYTHONPATH=vllm_plugin /workspace/.venv/bin/python benchmarks/transform_kernel_benchmark.py --output-json outputs/benchmark_results/kernels/a100.json
```

Expected: every accepted row passes the error gate and each shape/token pair has
multiple measured candidates.

- [ ] **Step 5: Implement only measured winners**

Add immutable configurations and bucket dispatch using the sweep medians. Pass
`num_warps` explicitly to the Triton launch. Preserve `BLOCK_N=64` and Triton's
default warp selection as the unknown-device/unknown-shape fallback.

- [ ] **Step 6: Validate kernel and end-to-end results**

Run transform tests, rerun the sweep for selected versus fallback, then repeat
both end-to-end workloads. Retain individual buckets only when kernel latency
improves and neither end-to-end workload regresses beyond noise.

- [ ] **Step 7: Commit Task 4**

```bash
git add benchmarks/transform_kernel_benchmark.py tests/test_triton_transform.py vllm_plugin/flatquant_vllm_plugin/triton_transform_v2.py
git commit -m "perf: tune FlatQuant transforms by EXAONE shape"
```

### Task 5: Safe Selective-Transform Export Experiment

**Files:**
- Create: `tests/test_export_flatquant_vllm.py`
- Modify: `tools/export_flatquant_vllm.py`

**Interfaces:**
- Produces: `TransformSelection` parsed from repeated `--exclude-transform` values.
- Produces: exported metadata listing every included and excluded projection/layer.

- [ ] **Step 1: Write failing selection and safety tests**

Cover projection families `qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj`,
inclusive layer ranges, invalid projection names, and the invariant that excluded
transforms require weights newly quantized without those transforms. Assert the
tool refuses to relabel an already transformed packed checkpoint.

- [ ] **Step 2: Verify RED**

```bash
/workspace/.venv/bin/python -m pytest tests/test_export_flatquant_vllm.py -q
```

Expected: failure because selection parsing and safety metadata do not exist.

- [ ] **Step 3: Implement selection metadata and fail-closed validation**

Add parsing and manifest generation without changing the default export. Require
an explicit untransformed/recalibrated source marker before any exclusion is
accepted. Never implement a runtime flag that merely skips transforms.

- [ ] **Step 4: Verify tests and default-export dry run**

Run the new tests and the exporter's `--help`. If a suitable untransformed source
checkpoint is unavailable, record the external artifact blocker and stop this
stage without fabricating quality results.

- [ ] **Step 5: Commit experiment tooling**

```bash
git add tests/test_export_flatquant_vllm.py tools/export_flatquant_vllm.py
git commit -m "feat: make selective FlatQuant exports fail closed"
```

### Task 6: Small-M Transform-Marlin Feasibility Gate

**Files:**
- Create: `experiments/marlin_fusion/README.md`

**Interfaces:**
- Produces: a go/no-go document with measured lower bound, expected memory-traffic saving, transform-recomputation analysis, and occupancy limits.

- [ ] **Step 1: Establish the attainable upper bound**

From Task 2 profiles, sum small-M transform kernel time, transformed-buffer write
time, and Marlin input-read time. This is the maximum removable latency; compare
it with total TPOT. Reject implementation if the bound is below measurement
noise or below 2% of TPOT.

- [ ] **Step 2: Map Marlin tile ownership and reuse**

Read the exact vLLM 0.24 Marlin source used by the installed environment. Record
M/N/K tile sizes, activation staging, CTA ownership, and whether transformed K
tiles can be reused across N tiles without cross-CTA communication.

- [ ] **Step 3: Write the feasibility decision**

Document `GO` only if a design avoids per-output-tile transform recomputation,
fits shared-memory/register limits on A100, and has a predicted benefit above the
2% TPOT gate. Otherwise document `NO-GO` with the measured reason and keep the
existing separate path.

- [ ] **Step 4: If GO, start a separate TDD prototype plan**

Do not modify production Marlin in this plan. Write a dedicated design and plan
that names the exact installed vLLM source files found in Step 2 and covers
`M <= 16`, correctness comparison against transform-plus-Marlin, fallback
dispatch, and benchmark gates. If NO-GO, commit only the decision document.

- [ ] **Step 5: Commit the feasibility result**

```bash
git add experiments/marlin_fusion/README.md
git commit -m "docs: evaluate FlatQuant Marlin fusion feasibility"
```

### Task 7: Final Verification and Report

**Files:**
- Modify: `VLLM_AWQ_FLATQUANT_W4A16_PATH.md`

**Interfaces:**
- Consumes: all retained commits and benchmark JSON.
- Produces: reproducible before/after table and explicit unresolved blockers.

- [ ] **Step 1: Run the full relevant test suite**

```bash
/workspace/.venv/bin/python -m pytest tests/test_vllm_awq_benchmark.py tests/test_triton_transform.py tests/test_export_flatquant_vllm.py -q
```

Expected: zero failures; CUDA tests must run rather than skip on this A100.

- [ ] **Step 2: Run fresh-process final benchmarks**

Run FlatQuant and AWQ for both target workloads with identical non-eager engine
arguments, two warm-ups, and at least seven measurements. Save raw JSON and full
logs. Generate identical fixed-prompt greedy outputs for the before/after
FlatQuant paths and compare token IDs.

- [ ] **Step 3: Run quality gates available on the local artifacts**

Run the repository perplexity path against the same dataset/config for baseline
and optimized FlatQuant. Run the text and vision smoke evaluations when their
datasets are locally available. Report unavailable datasets as blockers, not
passes.

- [ ] **Step 4: Update the performance document**

Record hardware/software versions, exact commands, TTFT, TPOT, throughput,
min/median/max, CUDA Graph state, correctness results, and rejected hypotheses.
Separate primary decode and secondary prefill conclusions.

- [ ] **Step 5: Verify documentation and repository state, then commit**

```bash
git diff --check
git status --short
git add VLLM_AWQ_FLATQUANT_W4A16_PATH.md
git commit -m "docs: report FlatQuant vLLM latency results"
```
