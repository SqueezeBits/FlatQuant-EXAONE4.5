# EXAONE-4.5-33B FlatQuant W4A4 vLLM Design

## Objective

Serve a genuinely weight-and-activation quantized EXAONE-4.5-33B model through
vLLM on one NVIDIA A100 80 GB GPU. The implementation must execute packed INT4
activations and packed INT4 weights on Tensor Cores. It must not use fake
quantization or silently run all linear layers through a weight-only path.

The primary optimization target is aggregate throughput for long prompts under
continuous batching. Batch-one decode latency is not a first-phase performance
target.

## Scope

The first phase supports:

- EXAONE-4.5-33B on one A100 (SM80)
- tensor parallel size 1
- W4A4 text-decoder projections
- BF16 embedding, normalization, language-model head, and vision encoder
- vLLM scheduling, continuous batching, chunked prefill, prefix caching, and
  paged attention without scheduler changes
- vLLM FP8 KV cache rather than FlatQuant KV4
- a separately exported FlatQuant W4A4 checkpoint

Tensor parallelism, H100-specific kernels, INT4 KV cache, and W4A4 vision
layers are explicitly deferred.

## Selected Integration Approach

Extend the out-of-tree vLLM quantization plugin. Register a dedicated
`flatquant_w4a4` quantization configuration and linear method while retaining
vLLM's native EXAONE model implementation.

This is preferable to copying the full EXAONE vLLM model because it limits the
maintenance surface to checkpoint loading, transforms, activation
quantization, and linear kernels. It is also preferable to forking vLLM for the
single-GPU first phase.

The existing W4A16 plugin remains a separate backend. W4A4 does not pretend to
be compressed-tensors WNA16/Marlin because that storage and execution contract
assumes floating-point activations.

## Checkpoint Format and Export

A new exporter converts a calibrated real-quantized FlatQuant checkpoint to a
vLLM-specific W4A4 directory. It emits:

- signed INT4 weights packed low nibble first in `uint8` tensors
- one FP16 weight scale per output channel
- projection-specific FlatQuant online-transform matrices
- learned activation clipping parameters or fixed clipping ratios
- model configuration and tokenizer assets
- `flatquant_w4a4_config.json`, including bit widths, packing version, target
  layers, excluded layers, transform metadata, and KV-cache policy

The exporter maps Hugging Face projection tensors to the fused vLLM layout:

- `q_proj`, `k_proj`, and `v_proj` become `qkv_proj`
- `gate_proj` and `up_proj` become `gate_up_proj`
- `o_proj` and `down_proj` remain separate

Packing and concatenation operate on unpacked logical output-channel ranges or
on independently packed row ranges; nibbles from different source projections
are never interleaved. Scales and transform metadata follow the same logical
output-channel ordering as the packed weights.

At load time the plugin validates the packing version, bit widths, dtype,
alignment, expected fused ranges, transform shapes, and required metadata. A
malformed or incomplete W4A4 checkpoint fails during model loading.

## Runtime Architecture

For each quantized projection, the linear method performs:

```text
BF16 input
  -> FlatQuant online transform
  -> per-token symmetric INT4 quantization and packing
  -> CUTLASS INT4 x INT4 GEMM with INT32 accumulation
  -> activation-scale x weight-scale application and optional bias
  -> BF16 output
```

The text decoder applies this path to `qkv_proj`, `o_proj`, `gate_up_proj`, and
`down_proj`. Other modules stay in their stated floating-point format.

The hot path does not construct `PackedQuantizedTensor` Python objects. Custom
operators accept ordinary tensors and return either packed activation plus
scale tensors or the final BF16 output.

### Kernel Boundaries

The initial correctness implementation may call three operations:

1. online transform plus activation quantization and INT4 packing
2. packed INT4 GEMM to INT32
3. scale application, optional bias, and BF16 conversion

The optimized implementation fuses the online transform, row-wise absolute-max
reduction, clipping, quantization, and packing. It also fuses scale application
and BF16 conversion into the GEMM epilogue. The intended intermediate layouts
are:

- activation: `[M, K / 2]` packed `uint8`
- activation scale: `[M, 1]` FP16
- weight: `[N, K / 2]` packed `uint8`
- weight scale: `[N, 1]` FP16
- accumulator: logical `[M, N]` INT32
- output: `[M, N]` BF16

The existing SM80 CUTLASS kernel is the reference starting point, not the final
performance target. Kernel configurations are tuned for EXAONE projection
shapes and representative prefill `M` ranges.

Temporary buffers are obtained through graph-safe reusable workspaces. The
forward path must not create shape-dependent Python objects or perform an
unbounded CUDA allocation after warm-up.

## vLLM Execution and KV Cache

vLLM retains ownership of request scheduling, continuous batching, paged
attention, and KV-cache allocation. The first phase uses vLLM's FP8 KV cache.
FlatQuant's existing KV4 kernels use a different page layout and are not
connected to vLLM until a separate paged-attention design is completed.

The normal serving configuration uses:

- tensor parallel size 1
- FP8 KV-cache dtype
- chunked prefill and prefix caching where supported by the selected vLLM
  version
- CUDA Graph only after eager-mode correctness passes

## Dispatch and Fallback Policy

Large prefill matrices use W4A4. Small decode matrices can be slower because
transform and quantization overhead dominates; therefore the backend supports
a measured `M` threshold below which it dispatches to a fallback.

The preferred fallback is the existing FlatQuant W4A16/Marlin method if the
checkpoint contains the required compatible representation. Otherwise a BF16
fallback may be enabled explicitly. The exporter does not duplicate W4A16 or
BF16 weights by default; a fallback-capable artifact must request and declare
the extra representation explicitly.

Production observability reports invocation counts by projection and dispatch
kind: W4A4, W4A16 fallback, and BF16 fallback. With
`FLATQUANT_W4A4_STRICT=1`, any dispatch from a targeted projection to a
fallback is an error. Accuracy gates and long-prefill performance measurements
run in strict mode so that fallback cannot inflate results.

## Error Handling

The backend rejects unsupported devices, tensor parallel sizes other than one,
unsupported bit widths, odd packed dimensions, incompatible alignment, missing
metadata, or unknown packing versions before inference. Runtime shape checks
include the layer prefix and expected and observed dimensions.

Kernel launch errors are surfaced immediately during correctness testing. The
serving process does not catch a kernel error and retry with floating point,
because doing so could conceal corrupted output or misleading performance.

## Verification

### Kernel Correctness

- Verify INT4 pack and unpack bit-for-bit for boundary values `[-8, 7]`.
- Compare activation scales, clipping, quantization, and packing with a PyTorch
  reference.
- Compare INT32 accumulators exactly with an unpacked integer matmul reference.
- Compare fused BF16 output against reference scale application under a stated
  numeric tolerance.
- Cover all EXAONE `qkv`, `o`, `gate_up`, and `down` shapes and representative
  prefill row counts.

### Model Correctness

- Compare layer outputs and logits against the existing Transformers FlatQuant
  real-W4A4 runtime using the same exported parameters.
- Compare deterministic greedy generations and report token agreement.
- Measure WikiText-2 perplexity and require it to stay within the tolerance
  established from repeated reference-runtime evaluation.
- Run strict mode and assert that every targeted projection uses W4A4.

Numeric tolerances and the acceptable perplexity delta are set from empirical
reference runs before declaring model correctness; they are recorded in the
test configuration rather than embedded as undocumented constants.

### Performance

Compare on the same A100 and software environment:

- EXAONE BF16 vLLM
- FlatQuant W4A16 vLLM
- FlatQuant W4A4 vLLM

Measure input lengths 2K, 8K, and 16K at concurrency 1, 4, 8, and 16 with short
outputs. Record prompt tokens per second, aggregate request throughput, time to
first token, peak GPU memory, and achieved concurrency. Use identical scheduler,
cache, compilation, and CUDA Graph settings for comparable runs.

The first phase succeeds when:

- W4A4 produces a reproducible aggregate prefill-throughput improvement over
  W4A16 for the selected long-prompt concurrency workloads
- W4A4 raises achievable concurrency or lowers peak memory without excessive
  workspace growth
- model correctness matches the reference runtime within the recorded gates
- strict-mode fallback count is zero for all performance results presented as
  W4A4

No fixed speedup percentage is required before measurements establish a stable
baseline. Results include raw values and variance rather than only a ratio.

## Implementation Sequence

1. Define the W4A4 config schema and exporter, including fused EXAONE mappings.
2. Add packing, loader, and schema unit tests.
3. Register the W4A4 plugin and graph-safe parameter objects.
4. Connect the existing SM80 W4A4 kernels as a correctness backend.
5. Establish kernel and end-to-end reference comparisons.
6. Fuse transform, activation quantization, and packing.
7. Fuse dequantization and BF16 conversion into the GEMM epilogue.
8. Tune kernels by EXAONE shape and representative prefill `M` buckets.
9. Add dispatch counters, strict mode, and measured small-`M` fallback.
10. Validate CUDA Graph operation and run the full accuracy and throughput
    matrix.

KV4, tensor parallelism, and additional GPU architectures require separate
designs after this phase meets its correctness and throughput gates.
