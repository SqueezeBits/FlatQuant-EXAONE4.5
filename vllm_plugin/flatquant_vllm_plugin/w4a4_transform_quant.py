"""Transform directly into row-scaled, low-nibble-first INT4 activations.

The two launches deliberately recompute the transform: the first reduces row
maxima, and the second quantizes and packs.  This removes the large transformed
activation buffer (and its read by the quantizer) while retaining the exact
bf16 rounding point of the normal transform path.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kron_tile(x, left, right, batch, pid_n, L: tl.constexpr, R: tl.constexpr,
               LP: tl.constexpr, RP: tl.constexpr, BN: tl.constexpr):
    il = tl.arange(0, LP)
    ir = tl.arange(0, RP)
    on = pid_n * BN + tl.arange(0, BN)
    xv = tl.load(x + (batch * L + il[:, None]) * R + ir[None, :],
                 mask=(il[:, None] < L) & (ir[None, :] < R), other=0.0)
    rv = tl.load(right + ir[:, None] * R + on[None, :],
                 mask=(ir[:, None] < R) & (on[None, :] < R), other=0.0)
    tmp = tl.dot(xv, rv)
    lt = tl.load(left + il[None, :] * L + il[:, None],
                 mask=(il[:, None] < L) & (il[None, :] < L), other=0.0)
    return tl.dot(lt.to(tmp.dtype), tmp), il, on


@triton.jit
def _left_tile(x, left, batch, pid_n, L: tl.constexpr, R: tl.constexpr,
               LP: tl.constexpr, BN: tl.constexpr):
    il = tl.arange(0, LP)
    on = pid_n * BN + tl.arange(0, BN)
    lt = tl.load(left + il[None, :] * L + il[:, None],
                 mask=(il[:, None] < L) & (il[None, :] < L), other=0.0)
    xv = tl.load(x + (batch * L + il[:, None]) * R + on[None, :],
                 mask=(il[:, None] < L) & (on[None, :] < R), other=0.0)
    return tl.dot(lt, xv), il, on


@triton.jit
def _row_max_kernel(x, left, right, maxima, L: tl.constexpr, R: tl.constexpr,
                    LP: tl.constexpr, RP: tl.constexpr, BN: tl.constexpr,
                    KRON: tl.constexpr, BF16: tl.constexpr):
    batch, pid_n = tl.program_id(0), tl.program_id(1)
    if KRON:
        out, il, on = _kron_tile(x, left, right, batch, pid_n, L, R, LP, RP, BN)
    else:
        out, il, on = _left_tile(x, left, batch, pid_n, L, R, LP, BN)
    if BF16:
        out = out.to(tl.bfloat16).to(tl.float32)
    else:
        out = out.to(tl.float16).to(tl.float32)
    valid = (il[:, None] < L) & (on[None, :] < R)
    tile_max = tl.max(tl.where(valid, tl.abs(out), 0.0))
    tl.atomic_max(maxima + batch, tile_max)


@triton.jit
def _pack_kernel(x, left, right, clip, maxima, packed, scales,
                 L: tl.constexpr, R: tl.constexpr, LP: tl.constexpr,
                 RP: tl.constexpr, BN: tl.constexpr, KRON: tl.constexpr,
                 BF16: tl.constexpr):
    batch, pid_n = tl.program_id(0), tl.program_id(1)
    # BN output columns are produced so adjacent nibbles never cross programs.
    if KRON:
        out, il, on = _kron_tile(x, left, right, batch, pid_n, L, R, LP, RP, BN)
    else:
        out, il, on = _left_tile(x, left, batch, pid_n, L, R, LP, BN)
    if BF16:
        out = out.to(tl.bfloat16).to(tl.float32)
    else:
        out = out.to(tl.float16).to(tl.float32)
    scale = tl.maximum(tl.load(maxima + batch) * tl.load(clip), 1.0e-8) * (1.0 / 7.0)
    q = libdevice.float2int_rn(libdevice.div_rn(out, scale))
    q = tl.maximum(-8, tl.minimum(q, 7))
    pair = tl.arange(0, BN // 2)
    even_n = pid_n * BN + pair * 2
    # Select adjacent columns from the tile without a global transformed tensor.
    q3 = tl.reshape(q, (LP, BN // 2, 2), can_reorder=False)
    lo, hi = tl.split(q3)
    byte = (lo & 0xF) | ((hi & 0xF) << 4)
    dst_col = il[:, None] * (R // 2) + even_n[None, :] // 2
    mask = (il[:, None] < L) & (even_n[None, :] < R)
    tl.store(packed + batch * (L * R // 2) + dst_col, byte.to(tl.uint8), mask=mask)
    if pid_n == 0:
        tl.store(scales + batch, scale)


def _run(x, left, right, clip):
    if not x.is_cuda:
        raise ValueError("transform_quantize_pack requires CUDA tensors")
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError("x must be a 2D tensor with even width")
    x = x.contiguous()
    left = left.contiguous()
    clip = clip.contiguous()
    kron = right is not None
    if kron:
        right = right.contiguous()
        r = right.shape[0]
        if x.shape[1] != left.shape[0] * r:
            raise ValueError("input width does not match transform factors")
    else:
        r = x.shape[1] // left.shape[0]
        right = left  # unused pointer; keeps one kernel signature
    rows, width = x.shape
    l = left.shape[0]
    lp, rp = triton.next_power_of_2(l), triton.next_power_of_2(r)
    bn = max(16, min(64, rp))
    if bn % 2:
        bn = 2
    packed = torch.empty((rows, width // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    maxima = torch.zeros((rows,), device=x.device, dtype=torch.float32)
    grid = (rows, triton.cdiv(r, bn))
    args = dict(L=l, R=r, LP=lp, RP=rp, BN=bn, KRON=kron,
                BF16=x.dtype == torch.bfloat16, num_warps=4)
    _row_max_kernel[grid](x, left, right, maxima, **args)
    _pack_kernel[grid](x, left, right, clip, maxima, packed, scales, **args)
    return packed, scales


_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")
if _HAS_CUSTOM_OP:
    @torch.library.custom_op("flatquant::kron_transform_quantize_pack", mutates_args=())
    def _kron_op(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
                 clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _run(x, left, right, clip)

    @_kron_op.register_fake
    def _(x, left, right, clip):
        return (torch.empty((x.shape[0], x.shape[1] // 2), device=x.device,
                            dtype=torch.uint8),
                torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32))

    @torch.library.custom_op("flatquant::left_transform_quantize_pack", mutates_args=())
    def _left_op(x: torch.Tensor, left: torch.Tensor,
                 clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _run(x, left, None, clip)

    @_left_op.register_fake
    def _(x, left, clip):
        return (torch.empty((x.shape[0], x.shape[1] // 2), device=x.device,
                            dtype=torch.uint8),
                torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32))


def transform_quantize_pack(x, left, right, clip):
    """Apply a left-only or Kronecker transform and return packed INT4 + scales."""
    x = x.contiguous()
    if _HAS_CUSTOM_OP and x.is_cuda:
        if right is None:
            return torch.ops.flatquant.left_transform_quantize_pack(x, left, clip)
        return torch.ops.flatquant.kron_transform_quantize_pack(x, left, right, clip)
    return _run(x, left, right, clip)
