# EXAONE 4.5 W4A4 A100 validation status

## Verified: tiny conditional W4A4 CUDA Graph path

On 2026-07-14, the Task 5 tiny `Exaone4_5_ForConditionalGeneration`
checkpoint was exported to the real packed W4A4 format and loaded by vLLM
0.24.0 on one NVIDIA A100-SXM4-80GB. PyTorch was 2.11.0+cu130.

The graph test used `enforce_eager=False`. vLLM compiled its dynamic `(1,
8192)` token range, captured 51 mixed prefill/decode piecewise graphs and 35
full decode graphs, then exercised prompt lengths 2 and 3 with decode lengths 1
and 2. Each case replayed alternating equal-shape prompts four times. Changed
values changed logits, repeated values reproduced logits, and
`torch.cuda.memory_allocated()` stayed at the post-warm-up baseline after every
replay. Native `LLM.generate()` output also returned the requested token count
and reproduced the same token IDs for repeated inputs. The W4A4 quantize, GEMM,
and transform operators all had symbolic fake/
Meta implementations. Model construction selected exactly four unique fused
W4A4 projection prefixes. This is selection evidence, not a replay call count.

Reproduction in this instance:

```bash
source /workspace/.venv/bin/activate
PYTHONPATH=.:vllm_plugin:third-party/fast-hadamard-transform \
  FLATQUANT_W4A4_STRICT=1 \
  pytest -q tests/test_exaone45_w4a4_logits.py -k cuda_graph
```

## Blocked: real 33B BF16/W4A16/W4A4 throughput matrix

No exported real EXAONE-4.5-33B W4A4 checkpoint is present. Consequently no
BF16/W4A16/W4A4 throughput rows, variance, OOM comparison, or empirical 33B
success claim is reported here. The matrix runner validates all three local
model directories and the W4A4 manifest before loading vLLM or writing JSON;
the missing W4A4 artifact exits nonzero with “the matrix was not run.” The tiny
fixture is never substituted for the 33B model.

When all real artifacts exist, run:

```bash
FLATQUANT_W4A4_STRICT=1 python benchmarks/exaone45/w4a4_throughput_matrix.py \
  --bf16-model /local/models/EXAONE-4.5-33B-bf16 \
  --w4a16-model outputs/EXAONE-4.5-33B/w4a16-vllm/exaone45-33b-w4a16-vllm \
  --w4a4-model outputs/EXAONE-4.5-33B/w4a4-vllm \
  --input-lengths 2048 8192 16384 \
  --concurrencies 1 4 8 16 \
  --output-length 32 --kv-cache-dtype fp8 \
  --json outputs/exaone45-w4a4-a100-throughput.json
```

Each backend/length/concurrency row runs in a fresh subprocess and records
prompt tokens/s, aggregate requests/s, median and p95 TTFT, peak allocated GPU
memory, completed requests, structured warm-up/measurement errors, and W4A4
model-construction selection evidence. The command exits zero and labels the
result `verified` only when every row completes all requested requests and all
W4A4 rows have positive selection evidence with zero fallback fields.

The final serving command, once the real artifact exists, is:

```bash
FLATQUANT_W4A4_STRICT=1 vllm serve \
  outputs/EXAONE-4.5-33B/w4a4-vllm \
  --quantization flatquant_w4a4 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8
```
