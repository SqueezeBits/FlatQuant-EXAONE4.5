# FlatQuant W4A16 for EXAONE 4.5

FlatQuant W4A16 inference and evaluation for
[LGAI-EXAONE/EXAONE-4.5-33B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B),
including a vLLM plugin backed by compressed-tensors/Marlin.

The pre-quantized checkpoint is available on Hugging Face:
[Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16](https://huggingface.co/Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16).

This repository is derived from the official
[ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant) implementation.
It is independently maintained by SqueezeBits and is not an official repository
of the original FlatQuant authors. See [LICENSE](LICENSE).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install "vllm==0.24.0"
pip install "git+https://github.com/SqueezeBits/FlatQuant-EXAONE4.5.git#subdirectory=vllm_plugin"
```

```bash
vllm serve \
  Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --dtype bfloat16 \
  --tensor-parallel-size 1
```

The current vLLM path supports weight-only FlatQuant W4A16 with BF16
activations and KV cache. Tensor parallelism is currently limited to TP=1.
See [EXAONE45_QUICKSTART.md](EXAONE45_QUICKSTART.md) for quantization, export,
and development instructions.

Export a calibrated, real-quantized checkpoint in the native W4A4 vLLM format:

```bash
python tools/export_flatquant_w4a4_vllm.py \
  --source outputs/EXAONE-4.5-33B/w4a4-real \
  --output outputs/EXAONE-4.5-33B/w4a4-vllm
```

### Strict W4A4 correctness gates

The correctness runner accepts local checkpoints only; it never downloads a
model or substitutes another architecture. Strict mode makes unsupported
shapes/backends fail instead of silently taking a fallback path:

```bash
FLATQUANT_W4A4_STRICT=1 python benchmarks/exaone45/vllm_w4a4.py generate \
  --model outputs/EXAONE-4.5-33B/w4a4-vllm \
  --prompt "Explain activation quantization in one paragraph." \
  --max-tokens 32 --enforce-eager --report w4a4-generate.json

FLATQUANT_W4A4_STRICT=1 python benchmarks/exaone45/vllm_w4a4.py logits \
  --model outputs/EXAONE-4.5-33B/w4a4-vllm \
  --reference outputs/EXAONE-4.5-33B/w4a4-real \
  --layer-tolerance <recorded-layer-max> \
  --logit-tolerance <recorded-logit-max> --min-token-agreement <recorded-min>

FLATQUANT_W4A4_STRICT=1 python benchmarks/exaone45/vllm_w4a4.py ppl \
  --model outputs/EXAONE-4.5-33B/w4a4-vllm --dataset wikitext2 \
  --dataset-path <local-wikitext-2-test.txt> \
  --ppl-min <recorded-min> --ppl-max <recorded-max>
```

The `logits` gate runs the repository's deploy-mode Transformers FlatQuant
adapter and vLLM with the same token IDs, compares full-vocabulary log
probabilities (vLLM's supported offline equivalent to raw logits), and enforces
the supplied recorded tolerances. The `ppl` gate
uses vLLM prompt log probabilities over a local WikiText-2 text artifact and
enforces the supplied recorded PPL interval. Neither command downloads data or
invents tolerances: absent 33B checkpoints, dataset, or recorded values fail
clearly, and no real-model result is claimed in this repository.

## Evaluation

Run these benchmarks from the `flatquant-vllm` environment described in
[EXAONE45_QUICKSTART.md](EXAONE45_QUICKSTART.md).

### Perplexity

```bash
python benchmarks/benchmark_exaone45.py ppl \
  --models bf16 awq flatquant \
  --awq_model_path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --flatquant_model_paths Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --datasets wikitext2 c4 --seqlen 2048 --max_samples 100000
```

| Dataset | Model | PPL | Avg. time (ms) | Samples |
| --- | --- | ---: | ---: | ---: |
| WikiText-2 | BF16 | 8.2628 | 591.36 | 131 |
| C4 | BF16 | 17.9321 | 571.48 | 221 |
| WikiText-2 | AWQ | 8.6353 | 752.57 | 131 |
| C4 | AWQ | 18.3595 | 749.31 | 221 |
| WikiText-2 | FlatQuant W4A16 | 8.3787 | 826.09 | 131 |
| C4 | FlatQuant W4A16 | 18.3465 | 885.84 | 221 |

### MMLU-Pro

Five-shot evaluation with lm-eval:

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models bf16 awq flatquant \
  --awq_model_path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --flatquant_model_paths Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --tasks mmlu-pro --num_fewshot 5 --batch_size 8 --max_length 8192
```

| Subject | BF16 | AWQ | FlatQuant W4A16 |
| --- | ---: | ---: | ---: |
| **Overall** | **71.96** | **68.40** | **70.02** |
| Biology | 84.10 | 81.73 | 81.73 |
| Business | 74.02 | 67.55 | 71.61 |
| Chemistry | 75.18 | 70.67 | 74.56 |
| Computer Science | 75.37 | 72.44 | 73.17 |
| Economics | 79.86 | 76.90 | 77.84 |
| Engineering | 50.77 | 49.54 | 49.12 |
| Health | 74.57 | 71.64 | 73.47 |
| History | 64.04 | 63.78 | 61.94 |
| Law | 46.59 | 44.41 | 46.59 |
| Math | 85.05 | 82.68 | 83.27 |
| Other | 68.61 | 64.07 | 65.15 |
| Philosophy | 71.74 | 68.34 | 69.14 |
| Physics | 78.06 | 69.75 | 74.06 |
| Psychology | 78.07 | 76.57 | 76.82 |

### MMMU-Pro

VLM evaluation with lmms-eval:

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models bf16 awq flatquant \
  --awq_model_path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --flatquant_model_paths Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --tasks mmmu_pro --batch_size 8 --max_new_tokens 512 --max_model_len 8192
```

| Model | MMMU-Pro |
| --- | --- |
| BF16 | standard=17.57 / vision=21.50 |
| AWQ | standard=17.98 / vision=20.93 |
| FlatQuant W4A16 | standard=18.90 / vision=20.06 |

### Speed up - A100

```bash
python benchmarks/benchmark_exaone45.py latency \
  --models awq flatquant \
  --awq_model_path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --flatquant_model_paths Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --batch_size 1 --prefill_seq_len 2048 --decode_steps 256 \
  --warmup_steps 2 --num_repeats 10
```

| Metric | AWQ | FlatQuant W4A16 | AWQ 대비 |
| --- | ---: | ---: | ---: |
| Prefill (tokens/s) | 86,311.6 | 82,260.2 | **-4.69%** |
| Decode (ms/step) | 17.104 | 18.108 | **+5.87%** |
| Decode (tokens/s) | 58.5 | 55.2 | **-5.54%** |
| End-to-end (ms) | 4,402.42 | 4,660.62 | **+5.86%** |

#### FlatQuant A100 transform optimization

FlatQuant W4A16 uses the same vLLM scheduler, attention backend, and Marlin
W4A16 GEMM path as AWQ, with learned online transforms added before the
quantized projections. The following optimizations were accumulated from the
initial vLLM implementation to the result above:

- removed the identity allocation and unnecessary right-transform kernel from
  `o_proj`;
- fused the right and left Kronecker factors into one Triton kernel, keeping the
  intermediate in registers and removing a BF16 temporary buffer and its
  global-memory traffic;
- registered the transforms as PyTorch custom ops so vLLM can retain
  `torch.compile` and FULL/PIECEWISE CUDA Graph capture without graph breaks;
- selected `BLOCK_N=16/32/64` from the projection shape and token-count bucket
  on A100, with the previous safe configuration retained as the fallback for
  unknown GPUs and shapes.

The checkpoint, packed INT4 weights, transform factors, and transform equations
are unchanged. The pre-tuning and final kernels produced identical WikiText-2
PPL in an 8 x 2,048-token smoke test (`6.911681770112198`) and byte-identical
32-token greedy output; all 108 transform shape/config checks stayed within
`5e-3` relative maximum error against the fp32 reference.

#### W4A4 decode dispatch policy

`FLATQUANT_W4A4_MIN_ROWS` controls the flattened-row threshold (default `1`),
and `FLATQUANT_W4A4_STRICT=1` rejects a selected fallback with an error that
identifies the layer prefix, `M`, and selection. Worker-observable counters have
a stable schema: `w4a4`, `w4a16_fallback`, and `bf16_fallback`.

Current exports explicitly declare `representations: ["w4a4"]` and contain no
duplicate W4A16 or BF16 projection weights. Consequently, setting a threshold
above one requests an unavailable fallback and is rejected; it never silently
dequantizes or fabricates another representation. The config loader likewise
rejects fallback declarations because this exporter/runtime does not implement
matching tensors and a load path yet.

An A100-SXM4-80GB microbenchmark used 10 warmups and 30 measured iterations for
each of M `1, 2, 4, 8, 16, 32, 64, 128`, comparing the same learned transform
plus native W4A4 (including activation quantize/pack), Marlin W4A16, or BF16
across all four fused EXAONE 32B projections. Down-projection W4A16 won through
M=64, and M=128 was the first sampled point where W4A4 won every projection.
Because 128 is the final requested sample, persistence beyond it was not
established; this is a sampled crossover, not a claimed stable production
threshold. The raw measurements are recorded separately in
`.superpowers/sdd/task-8-a100-crossover.json`; it is not encoded in today's
W4A4-only artifact, whose safe threshold remains 1.

## Acknowledgements

This work builds on [FlatQuant](https://github.com/ruikangliu/FlatQuant) and
[EXAONE 4.5](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B).

```bibtex
@article{sun2024flatquant,
  title={FlatQuant: Flatness Matters for LLM Quantization},
  author={Sun, Yuxuan and Liu, Ruikang and Bai, Haoli and Bao, Han and Zhao, Kang and Li, Yuening and Hu, Jiaxin and Yu, Xianzhi and Hou, Lu and Yuan, Chun and others},
  journal={arXiv preprint arXiv:2410.09426},
  year={2024}
}
```
