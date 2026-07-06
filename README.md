# FlatQuant W4A16 for EXAONE 4.5

FlatQuant W4A16 inference and evaluation for
[LGAI-EXAONE/EXAONE-4.5-33B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B),
including a vLLM plugin backed by compressed-tensors/Marlin.

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
  --models flatquant \
  --awq_model_path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --flatquant_model_paths Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --tasks mmlu-pro --num_fewshot 5 --batch_size 8 --max_length 4096
```

| Subject | AWQ | FlatQuant W4A16 | Delta |
| --- | ---: | ---: | ---: |
| **Overall** | **68.79** | **70.07** | **+1.28** |
| Biology | 81.17 | 80.33 | -0.84 |
| Business | 68.44 | 71.36 | +2.92 |
| Chemistry | 71.11 | 74.29 | +3.18 |
| Computer Science | 73.17 | 73.90 | +0.73 |
| Economics | 77.25 | 76.42 | -0.83 |
| Engineering | 48.71 | 49.95 | +1.24 |
| Health | 72.25 | 74.21 | +1.96 |
| History | 62.99 | 61.68 | -1.31 |
| Law | 46.05 | 46.78 | +0.73 |
| Math | 82.90 | 83.72 | +0.82 |
| Other | 65.04 | 65.04 | 0.00 |
| Philosophy | 68.54 | 70.14 | +1.60 |
| Physics | 70.52 | 74.67 | +4.15 |
| Psychology | 76.32 | 76.44 | +0.12 |

### MMMU-Pro

VLM evaluation with lmms-eval:

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models awq flatquant \
  --awq_model_path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --flatquant_model_paths Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --tasks mmmu_pro --batch_size 8 --max_new_tokens 512 --max_model_len 8192
```

### Latency

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

| Metric | AWQ | FlatQuant W4A16 | Change |
| --- | ---: | ---: | ---: |
| Prefill (tokens/s) | 79,036.3 | 81,854.7 | **+3.57%** |
| Decode (ms/step) | 17.151 | 18.526 | **+8.02%** |
| Decode (tokens/s) | 58.3 | 54.0 | **-7.38%** |
| End-to-end (ms) | 4,416.47 | 4,767.66 | **+7.95%** |

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
