"""Optimized Triton Kronecker transform for the vLLM plugin.

Two implementations live here:

* :func:`flatquant_kron_transform_two_stage` keeps FlatQuant's original
  right-then-left structure with two kernel launches and one temporary global
  buffer. It is retained as an A/B reference.
* :func:`flatquant_kron_transform` fuses both Kronecker factors into a single
  kernel launch. It keeps the ``X @ right`` intermediate in registers and
  immediately applies the left factor, so no temporary global buffer is
  allocated and only one kernel is launched per projection.
"""

import torch
import triton
import triton.language as tl

from .triton_transform import _right_transform_kernel, _torch_transform


@triton.jit
def _left_batched_kernel(
    x_ptr,
    left_ptr,
    out_ptr,
    left_size: tl.constexpr,
    right_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, left_size, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        left_t = tl.load(
            left_ptr + offs_k[None, :] * left_size + offs_m[:, None],
            mask=(offs_m[:, None] < left_size) & (offs_k[None, :] < left_size),
            other=0.0,
        )
        value = tl.load(
            x_ptr
            + (batch * left_size + offs_k[:, None]) * right_size
            + offs_n[None, :],
            mask=(offs_k[:, None] < left_size) & (offs_n[None, :] < right_size),
            other=0.0,
        )
        accumulator += tl.dot(left_t, value)
    tl.store(
        out_ptr
        + (batch * left_size + offs_m[:, None]) * right_size
        + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < left_size) & (offs_n[None, :] < right_size),
    )


@triton.jit
def _fused_kron_kernel(
    x_ptr,
    left_ptr,
    right_ptr,
    out_ptr,
    left_size: tl.constexpr,
    right_size: tl.constexpr,
    L_PAD: tl.constexpr,
    R_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute ``out[b] = left.T @ (x[b] @ right)`` for one batch/column tile.

    Each program owns one batch row and a ``BLOCK_N`` slice of the right
    dimension. It materialises the full ``L`` rows of the ``x @ right``
    intermediate in registers and folds the left factor into the same launch,
    so the two-stage temporary global buffer disappears.
    """
    batch = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_l = tl.arange(0, L_PAD)          # left index (output rows / left reduction)
    offs_r = tl.arange(0, R_PAD)          # right reduction index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns

    # x[batch] as [L_PAD, R_PAD] (rows over left, cols over right).
    x = tl.load(
        x_ptr + (batch * left_size + offs_l[:, None]) * right_size + offs_r[None, :],
        mask=(offs_l[:, None] < left_size) & (offs_r[None, :] < right_size),
        other=0.0,
    )
    # right[:, n_block] as [R_PAD, BLOCK_N].
    right = tl.load(
        right_ptr + offs_r[:, None] * right_size + offs_n[None, :],
        mask=(offs_r[:, None] < right_size) & (offs_n[None, :] < right_size),
        other=0.0,
    )
    # T = x @ right : [L_PAD, BLOCK_N], kept in registers (never written out).
    t = tl.dot(x, right)
    # left.T as [L_PAD, L_PAD] with A[m, k] == left[k, m].
    left_t = tl.load(
        left_ptr + offs_l[None, :] * left_size + offs_l[:, None],
        mask=(offs_l[:, None] < left_size) & (offs_l[None, :] < left_size),
        other=0.0,
    )
    # out = left.T @ T : [L_PAD, BLOCK_N].
    out = tl.dot(left_t.to(t.dtype), t)
    tl.store(
        out_ptr + (batch * left_size + offs_l[:, None]) * right_size + offs_n[None, :],
        out,
        mask=(offs_l[:, None] < left_size) & (offs_n[None, :] < right_size),
    )


def flatquant_kron_transform(x, left, right, *, backend="triton", block_n=64):
    """Fused single-launch ``x @ kron(left, right)`` preserving ``x.shape``."""
    if x.shape[-1] != left.shape[0] * right.shape[0]:
        raise ValueError("Input width does not match transform factors.")
    if backend == "torch" or not x.is_cuda:
        return _torch_transform(x, left, right)
    if backend != "triton":
        raise ValueError(f"Unknown backend: {backend}")

    shape = x.shape
    batches = x.numel() // x.shape[-1]
    left_size = left.shape[0]
    right_size = right.shape[0]
    l_pad = triton.next_power_of_2(left_size)
    r_pad = triton.next_power_of_2(right_size)
    block_n = min(block_n, r_pad)

    value = x.contiguous().reshape(batches * left_size, right_size)
    output = torch.empty_like(value)
    _fused_kron_kernel[(batches, triton.cdiv(right_size, block_n))](
        value,
        left.contiguous(),
        right.contiguous(),
        output,
        left_size=left_size,
        right_size=right_size,
        L_PAD=l_pad,
        R_PAD=r_pad,
        BLOCK_N=block_n,
    )
    return output.reshape(shape)


def flatquant_kron_transform_two_stage(x, left, right, *, backend="triton"):
    if x.shape[-1] != left.shape[0] * right.shape[0]:
        raise ValueError("Input width does not match transform factors.")
    if backend == "torch" or not x.is_cuda:
        return _torch_transform(x, left, right)
    if backend != "triton":
        raise ValueError(f"Unknown backend: {backend}")

    shape = x.shape
    batches = x.numel() // x.shape[-1]
    left_size = left.shape[0]
    right_size = right.shape[0]
    value = x.contiguous().reshape(batches * left_size, right_size)
    tmp = torch.empty_like(value)
    block = 16
    _right_transform_kernel[
        (triton.cdiv(value.shape[0], block), triton.cdiv(right_size, block))
    ](
        value,
        right.contiguous(),
        tmp,
        rows=value.shape[0],
        inner=right_size,
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_K=block,
    )
    output = torch.empty_like(tmp)
    _left_batched_kernel[
        (
            batches,
            triton.cdiv(left_size, block),
            triton.cdiv(right_size, block),
        )
    ](
        tmp,
        left.contiguous(),
        output,
        left_size=left_size,
        right_size=right_size,
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_K=block,
    )
    return output.reshape(shape)


def flatquant_left_transform(x, left, *, backend="triton"):
    """Apply only the left Kronecker factor.

    EXAONE's attention output transform has no learned right factor. Keeping
    this path separate avoids an identity allocation and one kernel launch.
    """
    if x.shape[-1] % left.shape[0] != 0:
        raise ValueError("Input width is not divisible by the left factor size.")
    if backend == "torch" or not x.is_cuda:
        shape = x.shape
        batches = x.numel() // x.shape[-1]
        left_size = left.shape[0]
        right_size = x.shape[-1] // left_size
        value = x.reshape(batches, left_size, right_size)
        return torch.matmul(left.T, value).reshape(shape)
    if backend != "triton":
        raise ValueError(f"Unknown backend: {backend}")

    shape = x.shape
    batches = x.numel() // x.shape[-1]
    left_size = left.shape[0]
    right_size = x.shape[-1] // left_size
    value = x.contiguous().reshape(batches * left_size, right_size)
    output = torch.empty_like(value)
    block = 16
    _left_batched_kernel[
        (
            batches,
            triton.cdiv(left_size, block),
            triton.cdiv(right_size, block),
        )
    ](
        value,
        left.contiguous(),
        output,
        left_size=left_size,
        right_size=right_size,
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_K=block,
    )
    return output.reshape(shape)


# ---------------------------------------------------------------------------
# torch custom ops
#
# vLLM compiles the model with Dynamo and captures piecewise CUDA graphs. A raw
# Triton launch inside the traced region either forces a graph break or, worse,
# specializes the dynamic token dimension and trips torch.compile's shape
# guards. Wrapping the transforms as ``torch.library`` custom ops makes them
# opaque to Dynamo: the tracer sees a single op with a shape-preserving fake
# implementation, and the real Triton launch runs (and is CUDA-graph captured)
# at execution time. The token dimension only feeds the launch grid, never a
# ``tl.constexpr``, so a captured graph stays valid for the batch size it was
# captured for.
# ---------------------------------------------------------------------------

_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("flatquant::kron_transform", mutates_args=())
    def kron_transform_op(
        x: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return flatquant_kron_transform(x, left, right)

    @kron_transform_op.register_fake
    def _(x, left, right):
        return torch.empty_like(x)

    @torch.library.custom_op("flatquant::left_transform", mutates_args=())
    def left_transform_op(x: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        return flatquant_left_transform(x, left)

    @left_transform_op.register_fake
    def _(x, left):
        return torch.empty_like(x)


def kron_transform(x, left, right):
    """Kronecker transform through the custom op when available."""
    if _HAS_CUSTOM_OP and x.is_cuda:
        return torch.ops.flatquant.kron_transform(x, left, right)
    return flatquant_kron_transform(x, left, right)


def left_transform(x, left):
    """Left-only transform through the custom op when available."""
    if _HAS_CUSTOM_OP and x.is_cuda:
        return torch.ops.flatquant.left_transform(x, left)
    return flatquant_left_transform(x, left)
