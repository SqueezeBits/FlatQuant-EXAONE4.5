# Final A100 README Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the two A100 comparisons with one AWQ-versus-latest-FlatQuant table and a concise cumulative optimization summary.

**Architecture:** Run both models through the same vLLM latency runner with batch size 1, prefill length 2,048, decode length 256, two warmups, and ten measured repeats. Use only fresh measurements in the final table, then summarize the execution-path optimizations without exposing obsolete FlatQuant baseline results.

**Tech Stack:** Python, vLLM 0.24.0, FlatQuant vLLM plugin, Markdown.

## Global Constraints

- Use the installed tuned plugin whose transform source hash matches the current worktree.
- Compare AWQ and FlatQuant in one benchmark invocation with identical workload settings.
- Keep the accuracy statement limited to the verified PPL smoke, greedy-token, and transform-sweep evidence.

---

### Task 1: Measure the final A100 comparison

**Files:**
- Read: `benchmarks/benchmark_exaone45.py`
- Read: `benchmarks/exaone45/latency.py`
- Output: `outputs/benchmark_results/final-a100/`

- [ ] Verify the installed plugin and GPU are ready.
- [ ] Run AWQ and tuned FlatQuant with `--batch_size 1 --prefill_seq_len 2048 --decode_steps 256 --warmup_steps 2 --num_repeats 10`.
- [ ] Read the saved JSON and independently calculate FlatQuant-versus-AWQ percentages.

### Task 2: Replace the A100 README section

**Files:**
- Modify: `README.md`

- [ ] Keep one final AWQ-versus-FlatQuant table using Task 1 results.
- [ ] Remove the FlatQuant-baseline-versus-tuned table and baseline-specific narrative.
- [ ] Add a short cumulative summary covering the `o_proj` simplification, fused Kronecker kernel, temporary-buffer removal, custom op/CUDA Graph integration, and SM80 shape-aware scheduling.
- [ ] Retain a concise, accurately scoped correctness statement.

### Task 3: Verify and commit

**Files:**
- Verify: `README.md`
- Test: `tests/test_export_flatquant_vllm.py`
- Test: `tests/test_vllm_awq_benchmark.py`
- Test: `tests/test_triton_transform.py`

- [ ] Run `git diff --check`.
- [ ] Run the 24 relevant tests.
- [ ] Confirm the worktree contains only the intended documentation and plan changes.
- [ ] Commit the completed documentation update.
