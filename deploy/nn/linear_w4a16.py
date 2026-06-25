import torch
import torch.nn as nn


class LinearW4A16(nn.Module):
    """BF16 activation x packed INT4 weight linear using PyTorch's CUDA kernel."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        *,
        group_size=128,
        inner_k_tiles=8,
        output_dtype=None,
    ):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in_features={in_features} must be divisible by group_size={group_size}.")
        if in_features % (inner_k_tiles * 16) != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by inner_k_tiles * 16 "
                f"({inner_k_tiles * 16})."
            )
        if out_features % 8 != 0:
            raise ValueError(f"out_features={out_features} must be divisible by 8.")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        self.output_dtype = output_dtype

        # Empty buffers avoid allocating a temporary FP16 copy while the model
        # skeleton is built. load_packed_weight() replaces them during loading.
        self.register_buffer("weight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("scales_and_zeros", torch.empty(0, dtype=torch.bfloat16))
        if bias:
            self.register_buffer("bias", torch.empty(0, dtype=output_dtype or torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, module, **kwargs):
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            output_dtype=module.weight.dtype,
            **kwargs,
        )

    @torch.no_grad()
    def load_packed_weight(self, packed_weight, scale, device):
        expected_shape = (self.out_features, self.in_features // 2)
        if tuple(packed_weight.shape) != expected_shape:
            raise ValueError(
                f"Packed weight shape {tuple(packed_weight.shape)} does not match {expected_shape}."
            )
        if tuple(scale.shape) not in {(self.out_features,), (self.out_features, 1)}:
            raise ValueError(
                f"Scale shape {tuple(scale.shape)} must be ({self.out_features},) "
                f"or ({self.out_features}, 1)."
            )
        if packed_weight.dtype != torch.uint8:
            raise TypeError(f"Packed weight must be uint8, got {packed_weight.dtype}.")
        if not str(device).startswith("cuda"):
            raise NotImplementedError("LinearW4A16 currently requires a CUDA device.")

        packed_weight = packed_weight.to(device=device, non_blocking=True)

        # FlatQuant stores low-nibble-first signed two's-complement INT4.
        # PyTorch int4pack expects high-nibble-first offset-binary INT4.
        packed_weight = (
            (packed_weight >> 4) | ((packed_weight & 0x0F) << 4)
        ) ^ 0x88
        self.weight = torch.ops.aten._convert_weight_to_int4pack(
            packed_weight.contiguous(),
            self.inner_k_tiles,
        )

        scale = scale.reshape(self.out_features).to(
            device=device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )
        groups = self.in_features // self.group_size
        scales_and_zeros = torch.empty(
            (groups, self.out_features, 2),
            device=device,
            dtype=torch.bfloat16,
        )
        scales_and_zeros[..., 0] = scale.unsqueeze(0)
        scales_and_zeros[..., 1].zero_()
        self.scales_and_zeros = scales_and_zeros

    def forward(self, x):
        if self.weight.numel() == 0 or self.scales_and_zeros.numel() == 0:
            raise RuntimeError("LinearW4A16 weight has not been loaded.")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, got {x.shape[-1]}."
            )

        input_dtype = x.dtype
        output_dtype = self.output_dtype or input_dtype
        output_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, self.in_features).to(torch.bfloat16)
        output = torch.ops.aten._weight_int4pack_mm(
            x,
            self.weight,
            self.group_size,
            self.scales_and_zeros,
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.reshape(output_shape).to(output_dtype)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, "
            f"inner_k_tiles={self.inner_k_tiles}"
        )
