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
