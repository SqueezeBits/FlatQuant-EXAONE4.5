"""Symbolic fake implementations shared by deploy and the vLLM plugin."""

import torch


@torch.library.register_fake("flatquant::quantize_pack_i4")
def quantize_pack_i4_fake(x, clip):
    if x.dtype != torch.bfloat16:
        raise RuntimeError("W4A4 activation must be BF16")
    if clip.dtype != torch.float16 or clip.ndim != 1 or clip.shape[0] != 1:
        raise RuntimeError("clip must have shape (1,) and dtype FP16")
    if x.ndim != 2 or x.shape[1] % 64:
        raise RuntimeError("W4A4 requires 2D x and K % 64 == 0")
    if not x.is_contiguous() or not clip.is_contiguous():
        raise RuntimeError("inputs must be contiguous")
    return (
        torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8),
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
    )


@torch.library.register_fake("flatquant::w4a4_linear")
def w4a4_linear_fake(packed_x, packed_w, x_scale, w_scale, output_dtype, bias=None):
    if packed_x.dtype != torch.uint8 or packed_w.dtype != torch.uint8:
        raise RuntimeError("packed tensors must be uint8")
    if packed_x.ndim != 2 or packed_w.ndim != 2 or packed_x.shape[1] != packed_w.shape[1]:
        raise RuntimeError("packed tensors must be 2D with matching K")
    if packed_x.shape[1] % 32:
        raise RuntimeError("W4A4 requires K % 64 == 0")
    if packed_w.shape[0] % 8:
        raise RuntimeError("W4A4 requires N % 8 == 0")
    if output_dtype != torch.bfloat16:
        label = "output_dtype" if output_dtype == torch.float32 else "BF16"
        raise RuntimeError(f"{label} must be BF16")
    return torch.empty(
        (packed_x.shape[0], packed_w.shape[0]), device=packed_x.device, dtype=output_dtype
    )
