# FlatQuant transform–Marlin fusion feasibility (A100, vLLM 0.24.0)

## Decision

**NO-GO for an in-place Marlin fusion prototype.** Keep the transformed
activation materialized once per projection and continue optimizing the
standalone transform kernel. A different persistent GEMM design could revisit
this decision, but adding the transform to the current Marlin CTA is expected to
repeat more work than it removes.

This is an architecture decision, not a claim that removing transforms has no
value. The measured upper bound is large enough: FlatQuant's baseline decode
TPOT was 15.668 ms versus AWQ's 14.266 ms (1.402 ms, 9.8%), and the steady-state
profile assigned about 9.4% of summed CUDA-kernel time to FlatQuant transforms.
The shape-tuned standalone kernels reduced FlatQuant TPOT to 15.267 ms, leaving
about 1.001 ms (7.0%) relative to AWQ.

## Source evidence

The installed runtime is vLLM 0.24.0. Its matching source tag is commit
`ee0da84ab9e04ac7610e28580af62c365e898389`.

- `marlin.cuh` fixes a minimum N tile of 64, maximum N tile of 256, minimum K
  tile of 64, 256 threads, and four shared-memory pipeline stages.
- `marlin_template.h` computes `n_tiles = prob_n / (16 * thread_n_blocks)` and
  maps each threadblock to a stripe containing one or more N-column slices.
  The same M-row activation is therefore consumed independently by many
  N-owning CTAs.
- The kernel's A staging is local to each CTA's dynamic shared memory. Its
  global locks coordinate partial K reductions of an output slice; they do not
  provide cross-CTA sharing of a transformed A tile.
- `marlin.cu` launches `sms * blocks_per_sm` blocks and chooses N/K tiles to
  maximize SM utilization. On SM80 there is no threadblock-cluster distributed
  shared memory through which one CTA could transform A for its N neighbors.

For EXAONE-4.5-33B, K is 5,120 for attention/MLP input projections and 27,392
for `down_proj`. Since every N slice needs the full transformed K vector, placing
the Kronecker transform inside Marlin repeats it approximately once per active
N slice (at least one per 64–256 output columns), instead of once per projection.
The transform also needs values across K blocks, whereas Marlin intentionally
streams K tiles through a four-stage pipeline. Computing the transform in that
pipeline would either require global intermediates/synchronization or destroy
the current streaming schedule.

## Revisit criteria

Reconsider fusion only if one of these becomes available:

1. a persistent kernel assigns all relevant N tiles for an M row to one CTA or
   cooperative CTA cluster and demonstrably reuses transformed A;
2. a transform representation becomes K-tile-local, eliminating cross-tile
   dependencies; or
3. a newer GPU/backend offers cluster shared memory and a benchmark shows at
   least 2% end-to-end TPOT improvement without occupancy or quality regression.

Until then, modifying Marlin has a high maintenance cost (custom vLLM C++/CUDA
build and CUDA-graph validation) with a negative expected compute tradeoff.
