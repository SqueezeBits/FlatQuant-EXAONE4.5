"""Triton implementation of FlatQuant's decomposed Kronecker transform."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Allows checkpoint tooling to import on CPU-only hosts.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _right_transform_kernel(
        x_ptr,
        right_ptr,
        out_ptr,
        rows: tl.constexpr,
        inner: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, inner, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * inner + offs_k[None, :],
                mask=(offs_m[:, None] < rows) & (offs_k[None, :] < inner),
                other=0.0,
            )
            right = tl.load(
                right_ptr + offs_k[:, None] * inner + offs_n[None, :],
                mask=(offs_k[:, None] < inner) & (offs_n[None, :] < inner),
                other=0.0,
            )
            accumulator += tl.dot(x, right)
        tl.store(
            out_ptr + offs_m[:, None] * inner + offs_n[None, :],
            accumulator,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < inner),
        )


    @triton.jit
    def _left_transform_kernel(
        x_ptr,
        left_ptr,
        out_ptr,
        batches: tl.constexpr,
        left_size: tl.constexpr,
        right_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        total_rows = batches * left_size
        offs_flat_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        batch = offs_flat_m // left_size
        out_left = offs_flat_m % left_size
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, left_size, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            # left.T[out_left, k] == left[k, out_left]
            left = tl.load(
                left_ptr + offs_k[None, :] * left_size + out_left[:, None],
                mask=(offs_flat_m[:, None] < total_rows)
                & (offs_k[None, :] < left_size),
                other=0.0,
            )
            x = tl.load(
                x_ptr
                + (batch[:, None] * left_size + offs_k[None, :])
                * right_size
                + offs_n[:, None] * 0,
                mask=False,
                other=0.0,
            )
            # Load [K, N] separately; broadcasting batch across each output row.
            x = tl.load(
                x_ptr
                + (batch[:, None, None] * left_size + offs_k[None, :, None])
                * right_size
                + offs_n[None, None, :],
                mask=(offs_flat_m[:, None, None] < total_rows)
                & (offs_k[None, :, None] < left_size)
                & (offs_n[None, None, :] < right_size),
                other=0.0,
            )
            # Each BLOCK_M row can have a different batch, so reduce explicitly.
            accumulator += tl.sum(left[:, :, None] * x, axis=1)
        tl.store(
            out_ptr + offs_flat_m[:, None] * right_size + offs_n[None, :],
            accumulator,
            mask=(offs_flat_m[:, None] < total_rows)
            & (offs_n[None, :] < right_size),
        )


def _torch_transform(x, left, right):
    shape = x.shape
    value = x.reshape(-1, left.shape[0], right.shape[0])
    value = torch.matmul(value, right)
    value = torch.matmul(left.T, value)
    return value.reshape(shape)


def flatquant_kron_transform(x, left, right, *, backend="triton"):
    """Apply ``x @ kron(left, right)`` while preserving ``x.shape``.

    ``backend='torch'`` is the correctness reference and CPU fallback.
    """
    if x.shape[-1] != left.shape[0] * right.shape[0]:
        raise ValueError(
            f"Input width {x.shape[-1]} does not match transform factors "
            f"{left.shape[0]} * {right.shape[0]}."
        )
    if left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("left must be square.")
    if right.ndim != 2 or right.shape[0] != right.shape[1]:
        raise ValueError("right must be square.")
    if backend == "torch" or not x.is_cuda:
        return _torch_transform(x, left, right)
    if backend != "triton":
        raise ValueError(f"Unknown backend: {backend}")
    if triton is None:
        raise RuntimeError("Triton is required for backend='triton'.")

    original_shape = x.shape
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
    _left_transform_kernel[
        (triton.cdiv(batches * left_size, block), triton.cdiv(right_size, block))
    ](
        tmp,
        left.contiguous(),
        output,
        batches=batches,
        left_size=left_size,
        right_size=right_size,
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_K=block,
    )
    return output.reshape(original_shape)
