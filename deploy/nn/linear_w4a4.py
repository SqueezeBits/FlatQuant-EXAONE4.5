import torch


def _require_sm80(device):
    if torch.device(device).type != "cuda":
        raise NotImplementedError("packed W4A4 currently supports CUDA SM80 only")
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise NotImplementedError("packed W4A4 currently supports NVIDIA SM80 only")


def quantize_pack_i4(x: torch.Tensor, clip: torch.Tensor):
    _require_sm80(x.device)
    if x.dtype != torch.bfloat16:
        raise TypeError(f"W4A4 activation must be BF16, got {x.dtype}")
    if x.ndim != 2 or x.shape[1] % 64:
        raise ValueError("W4A4 requires a 2D activation and K % 64 == 0")
    if clip.shape != (1,) or clip.dtype != torch.float16:
        raise ValueError("activation clip must have shape (1,) and dtype float16")
    return torch.ops.flatquant.quantize_pack_i4(x.contiguous(), clip.contiguous())


def w4a4_linear(
    packed_x: torch.Tensor,
    packed_w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype=torch.bfloat16,
):
    if packed_x.ndim != 2 or packed_w.ndim != 2:
        raise ValueError("packed_x and packed_w must be 2D")
    if packed_x.shape[1] != packed_w.shape[1]:
        raise ValueError("packed activation and weight K mismatch")
    if x_scale.shape != (packed_x.shape[0], 1):
        raise ValueError("x_scale shape mismatch")
    if w_scale.shape != (packed_w.shape[0], 1):
        raise ValueError("w_scale shape mismatch")
    _require_sm80(packed_x.device)
    output = torch.ops.flatquant.w4a4_linear(
        packed_x.contiguous(), packed_w.contiguous(), x_scale.contiguous(),
        w_scale.contiguous(), output_dtype
    )
    return output if bias is None else output + bias


class LinearW4A4(torch.nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, *, output_dtype=torch.bfloat16
    ):
        super().__init__()
        if in_features % 64 or out_features % 8:
            raise ValueError("W4A4 requires K % 64 == 0 and N % 8 == 0")
        self.in_features = in_features
        self.out_features = out_features
        self.output_dtype = output_dtype
        self.register_buffer(
            "weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float16)
        )
        self.register_buffer("activation_clip", torch.ones(1, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=output_dtype))
        else:
            self.bias = None
        self._weight_loaded = False

    @torch.no_grad()
    def load_packed_weight(self, weight, scale, device):
        if weight.shape != self.weight.shape or scale.shape != self.weight_scale.shape:
            raise ValueError("packed weight or scale shape mismatch")
        if weight.dtype != torch.uint8:
            raise TypeError(f"packed weight must be uint8, got {weight.dtype}")
        _require_sm80(device)
        self.weight = weight.to(device=device, non_blocking=True).contiguous()
        self.weight_scale = scale.to(
            device=device, dtype=torch.float16, non_blocking=True
        ).contiguous()
        self._weight_loaded = True

    def forward(self, x: torch.Tensor):
        if not self._weight_loaded:
            raise RuntimeError("LinearW4A4 weight has not been loaded")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"W4A4 activation must be BF16, got {x.dtype}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input feature dimension {self.in_features}, got {x.shape[-1]}"
            )
        output_shape = x.shape[:-1] + (self.out_features,)
        flat_x = x.reshape(-1, self.in_features)
        packed_x, x_scale = quantize_pack_i4(flat_x, self.activation_clip)
        output = w4a4_linear(
            packed_x, self.weight, x_scale, self.weight_scale,
            bias=self.bias, output_dtype=self.output_dtype
        )
        return output.reshape(output_shape)
