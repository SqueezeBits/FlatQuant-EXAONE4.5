# FlatQuant W4A16 vLLM Latency Optimization Design

## Objective

Reduce EXAONE-4.5 FlatQuant W4A16 latency on vLLM without silently changing the
model's numerical behavior. Optimize the measured production proxies in this
order:

1. Decode latency: batch 1, 64 input tokens, 128 output tokens.
2. Prefill/throughput: batch 4, 512 input tokens, 16 output tokens.

Compare every end-to-end result with both the unmodified FlatQuant baseline and
an AWQ W4A16 model under identical vLLM settings.

## Constraints

- Preserve the existing FlatQuant checkpoint path and default behavior.
- Do not remove a learned transform from an existing checkpoint; transformed
  weights and activations must remain in the same coordinate system.
- Accept an optimization only after an isolated A/B measurement shows an
  improvement in its target workload.
- Validate transform output against the FP32 PyTorch reference with relative
  maximum error no worse than the existing `5e-3` threshold.
- Validate end-to-end greedy output and perplexity before declaring a runtime
  optimization safe.
- Keep eager mode only as a diagnostic baseline. Production comparisons use
  vLLM compilation and CUDA Graphs when capture succeeds.
- Do not install or modify the NVIDIA driver.

## Measurement Design

Create one reproducible benchmark entry point that records:

- exact model and vLLM configuration;
- GPU, CUDA, PyTorch, Triton, and vLLM versions;
- TTFT, TPOT, end-to-end latency, input throughput, and output throughput;
- warm-up count, measured iterations, token counts, and batch size;
- whether compilation and CUDA Graph capture were active;
- median and dispersion rather than a single timing observation.

Run the FlatQuant and AWQ cases in separate processes to avoid allocator,
compilation-cache, and model-residency contamination. Use fixed prompts or fixed
random seeds and greedy decoding. Save machine-readable JSON results under
`outputs/benchmark_results/`, which remains untracked runtime output.

Before changing code, gather a baseline with Nsight Systems when available and
PyTorch/vLLM profiling otherwise. Attribute time separately to FlatQuant
transforms, Marlin GEMMs, attention, graph launches, copies, and allocations.

## Optimization Stages

### Stage 1: Confirm the Runtime Fast Path

Verify that production-style runs do not use `--enforce-eager`, that model
compilation completes, and that decode CUDA Graph capture includes the selected
batch sizes. Treat graph breaks, recompilations, or uncaptured decode as the
first root cause. No kernel change is justified until this fast path is proven.

### Stage 2: Remove Avoidable Copies and Dispatch Work

Instrument whether `x.contiguous()`, `left.contiguous()`, and
`right.contiguous()` cause real copies. Static transform factors should be laid
out once after loading. Activation handling should avoid a copy only when the
incoming strides make direct access correct. Preserve a safe fallback for an
unexpected non-contiguous activation.

Measure allocation and copy counts before and after. Keep the change only if it
improves at least one target workload without regressing the other by more than
noise.

### Stage 3: Shape-Aware Triton Tuning

Benchmark transform kernels independently for token buckets
`1, 2, 4, 8, 16, 64, 256, 512, 2048` and the EXAONE projection factor shapes.
Search a bounded set of block sizes and warp counts. Select configurations by
static factor shape and token bucket so CUDA Graph shapes remain stable.

Use Nsight Compute when available to reject configurations that spill registers
or materially reduce occupancy. Autotuning must have deterministic fallback
configurations and must not run an unbounded search during serving startup.

### Stage 4: Selective-Transform Checkpoint Experiment

This is an offline checkpoint experiment, not a runtime switch. Recalibrate and
requantize variants that omit transforms by projection family or by layer group.
Candidate order:

1. Remove `o_proj` transforms.
2. Remove attention-side transforms while retaining MLP transforms.
3. Remove MLP-side transforms while retaining attention transforms.
4. Retain transforms only for layers with the largest direct-INT4 error.

Evaluate latency and quality for each independently. Never load transformed
weights while disabling their matching online transform. Promote a variant only
if its quality/latency Pareto point is better than full FlatQuant and AWQ.

### Stage 5: Decode-Only Transform plus Marlin Prototype

Prototype fusion only for small-M decode shapes, initially `M <= 16`. The fused
kernel must avoid writing the complete transformed activation to global memory
and must reuse transformed K tiles across output tiles. If the design recomputes
the transform for every output tile, reject it before broad implementation.

Keep the current separate fused-transform-plus-Marlin path for prefill and as a
fallback. Advance beyond a prototype only when kernel and end-to-end measurements
both show a gain and register/shared-memory pressure remains acceptable.

## Validation Gates

Each runtime stage must pass:

1. Unit comparison against the FP32 transform reference across representative
   token counts, dtypes, and projection shapes.
2. Existing FlatQuant plugin and vision tests.
3. Fixed-prompt greedy token equality against the pre-change runtime.
4. End-to-end A/B latency in fresh processes for both target workloads.

Before selecting a checkpoint variant or fused GEMM path, additionally run the
existing perplexity and downstream text/vision evaluation paths. Record any
quality delta rather than treating successful generation as sufficient.

## Decision Rules

- Diagnose one bottleneck and test one hypothesis at a time.
- Retain a change only when repeated median latency improves beyond measurement
  noise; report confidence intervals or min/median/max when sample counts are
  small.
- If three kernel hypotheses fail, stop modifying that kernel and reconsider the
  architecture before another attempt.
- Prefer the lowest-complexity change that reaches the same measured result.
- Report improvements separately for TTFT, TPOT, and throughput; do not collapse
  them into one token/s number.

## Deliverables

- Reproducible AWQ versus FlatQuant benchmark command and JSON result schema.
- Baseline profile identifying the actual residual latency contributors.
- Copy/layout optimization if profiling proves copies occur.
- Shape-aware transform configurations with unit and kernel benchmarks.
- Selective-transform experiment tooling and quality/latency report.
- Small-M transform-plus-Marlin feasibility result, including a documented
  rejection if fusion is slower or structurally unsuitable.

