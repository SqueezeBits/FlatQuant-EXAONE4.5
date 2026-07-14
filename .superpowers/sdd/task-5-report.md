# Task 5 report: EXAONE-4.5 W4A4 correctness gates

Status: **NEEDS_CONTEXT (real-model gates blocked); deterministic harness GREEN**

## RED

`PYTHONPATH=$PWD/vllm_plugin:$PWD /venv/main/bin/pytest -q tests/test_exaone45_w4a4_logits.py`
failed during collection with
`ModuleNotFoundError: No module named 'benchmarks.exaone45.vllm_w4a4'`.
This was the expected failure before the comparison/report module existed.

## GREEN (executed)

The same focused test command passed: `5 passed in 2.87s`. It covers exact
mean/max deltas, token agreement, counter preservation, malformed input,
JSON-report shape, and local-only artifact failure. No model result is mocked.

Environment inspection found vLLM 0.24.0, Transformers 5.13.1,
PyTorch 2.11.0+cu130, and an A100-SXM4-80GB. vLLM registers both
`Exaone4ForCausalLM` and `Exaone4_5_ForConditionalGeneration`. The latter is
the required native route because it creates the text model under the
`language_model` prefix targeted by this W4A4 plugin.

## Real-model gates (executed only as missing-artifact checks)

The three prescribed `generate`, `logits`, and `ppl` invocations were executed
with `FLATQUANT_W4A4_STRICT=1`. Each exited 2 before engine construction with:

`model local checkpoint path does not exist: outputs/EXAONE-4.5-33B/w4a4-vllm`

Therefore no 33B generation, logits comparison, fallback count, tolerance, or
WikiText-2 PPL was measured. None is claimed.

## Blocking context

- `outputs/EXAONE-4.5-33B/w4a4-vllm` and
  `outputs/EXAONE-4.5-33B/w4a4-real` are absent.
- The repository does not contain the real-W4A4 Transformers reference adapter,
  WikiText-2 tokenization artifacts, measured layer/logit tolerances, or a
  measured reference PPL range. The `logits` and `ppl` subcommands validate
  paths and then stop clearly until that context is supplied.
- In vLLM 0.24.0 the engine workers own the plugin's in-process dispatch
  counter. The driver does not expose that worker-local count. Strict execution
  has no fallback branch, so a completed generation can truthfully record
  fallback=0, but the exact expected projection-call total cannot be proven
  through the public `LLM` interface yet; reports use `w4a4_calls: null`.
- A full tiny conditional checkpoint also needs the model-specific
  `Exaone4_5_Processor` files and exact HF-to-vLLM multimodal weight mapping.
  Creating a substitute text-only model would bypass the plugin prefix and
  would be a fake test, so it was not done.

To finish the task, supply the two real local checkpoints/reference artifacts
and either expose the worker counter through vLLM worker RPC or define an
approved worker-extension hook for the tiny fixture.
