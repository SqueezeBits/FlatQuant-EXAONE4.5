# Task 7 report: fused W4A4 epilogue and SM80 shape selection

## Scope and source dimensions

CUTLASS is v3.9.2. Its SM80 2.x epilogue visitor API supports both
`VisitorColBroadcast` and `VisitorRowBroadcast`. The implemented graph is INT32
accumulator fetch -> FP32 row-scale multiply -> FP32 column-scale multiply ->
BF16 visitor store. There is no `[M,N]` INT32 PyTorch tensor.

The requested exported model config was absent. Dimensions were derived from
the official cached EXAONE config at
`/workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/31e6a965d0661bbe4a8b895e22a77f8271772ba0/config.json`:
hidden size 5120, intermediate size 27392, 40 attention heads, and 8 KV heads.
Head size is 5120/40=128. The four unique `(N,K)` families are `(5120,5120)`,
`(1024,5120)`, `(27392,5120)`, and `(5120,27392)`.

## RED

`pytest -q tests/test_linear_w4a4.py -k 'epilogue or exaone'` against the old
separate epilogue produced 4 passes and 1 expected failure. For M=512, N=5120,
allocator peak delta was 15,728,640 bytes; the reliable fused threshold was
10,485,760 bytes. This is exactly BF16 output plus an additional 4-byte INT32
matrix.

## GREEN and build

- SM80-only extension build: successful. CUTLASS headers required explicit core,
  GEMM, numeric conversion, memory, and thread-map includes. A second compiler
  issue was caused by importing CUTE's internal anonymous namespace; fully
  qualifying CUTE names fixed the cudafe ambiguity.
- Targeted: `5 passed, 24 deselected`.
- Full W4A4 suite after selector and Meta parity coverage: `31 passed`.
- The allocator test now observes a 5,242,880-byte peak delta, exactly the
  5,242,880-byte BF16 output, versus the RED 15,728,640-byte peak.

## A100 benchmark

Command used the exact cached config source above and rows
`1 16 128 512 2048 8192`. JSON is `outputs/w4a4-gemm-a100.json` and contains 24
records. GPU: NVIDIA A100-SXM4-80GB. Maximum absolute error is 0.015455.
Effective TOPS are computed from measured time, not presented as hardware
claims. Every JSON record includes raw medians for all three candidates and its
measured winner; the final run's winner counts were 6/8/10 for candidates
0/1/2.

Candidate tuning used 30 measured samples after five warmups for all three
tiles, all four real shapes, and bucket representatives M=1,32,128,512,2048.
Raw tuning data is `outputs/w4a4-gemm-a100-candidates.json`. Candidate IDs are:

0. `64x128x128 / 32x64x128 / 3-stage`
1. `128x128x128 / 64x64x128 / 3-stage`
2. `128x256x128 / 64x64x128 / 3-stage`

The measured 4-by-5 winner table is encoded directly in `candidate_for`; unknown
aligned shapes retain a deterministic bucket fallback. The private
`_w4a4_linear_candidate` debug op validates candidate IDs and uses the identical
numeric path; production `w4a4_linear` ABI remains unchanged and BF16-only.

## Review correction

The first tuning results are invalid: the candidate switch incorrectly routed
IDs 0 and 1 to Small, ID 2 to Medium, and the validated default arm to Large.
The corrected exhaustive switch maps 0=Small, 1=Medium, 2=Large. Observable
candidate-name and full bucket-boundary tests now prove dispatch identity.

Corrected tuning gives each candidate five private warmups, rotates timing order
deterministically per shape, and takes 30 samples across all 3x4x5 combinations.
Corrected winners are: `(1024,5120)` all Small; `(5120,5120)`
Small/Small/Small/Small/Medium; `(27392,5120)`
Small/Small/Medium/Medium/Medium; `(5120,27392)`
Small/Small/Small/Medium/Medium. Corrected raw records replace
`outputs/w4a4-gemm-a100-candidates.json`.

Optional contiguous BF16 `(N,)` bias is a CUTLASS row-broadcast visitor and FP32
add before the single BF16 store. The schema accepts `Tensor? bias` with a
default, Meta and CUDA validate it, and the high-level module no longer performs
a separate `output + bias`. Bias memory verification observes exactly one BF16
output allocation.
