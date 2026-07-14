import torch


def pack_signed_i4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.int8 or values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("signed INT4 input must be a 2D int8 tensor with even K")
    if values.numel() and (values.min().item() < -8 or values.max().item() > 7):
        raise ValueError("signed INT4 values must be in [-8, 7]")
    encoded = torch.where(values < 0, values + 16, values).to(torch.uint8)
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).contiguous()


def unpack_signed_i4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed INT4 input must be a 2D uint8 tensor")
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    output = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int8)
    output[:, 0::2], output[:, 1::2] = low, high
    return output


def merge_output_rows(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts or any(part.dtype != torch.uint8 or part.ndim != 2 for part in parts):
        raise ValueError("fused projections require non-empty 2D uint8 tensors")
    if len({part.shape[1] for part in parts}) != 1:
        raise ValueError("fused projections must have the same packed K")
    return torch.cat(parts, dim=0).contiguous()
