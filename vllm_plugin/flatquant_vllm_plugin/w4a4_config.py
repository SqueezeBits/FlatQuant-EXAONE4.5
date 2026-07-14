"""Native packed W4A4 quantization configuration for EXAONE 4.5."""

from dataclasses import dataclass
import os
from typing import Any

import torch
from torch import nn

from flatquant_w4a4.format import validate_manifest
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from .transform import decompose_dim
from .w4a4_ops import apply_w4a4


@dataclass(frozen=True)
class DispatchPolicy:
    """Select a declared projection representation by flattened row count."""

    min_w4a4_rows: int
    strict: bool

    def __post_init__(self):
        if self.min_w4a4_rows < 1:
            raise ValueError("FLATQUANT_W4A4_MIN_ROWS must be at least 1")

    @classmethod
    def from_env(cls):
        return cls(
            min_w4a4_rows=int(os.getenv("FLATQUANT_W4A4_MIN_ROWS", "1")),
            strict=os.getenv("FLATQUANT_W4A4_STRICT", "0") == "1",
        )

    def select(self, rows: int, prefix: str, representations) -> str:
        if rows >= self.min_w4a4_rows:
            return "w4a4"
        available = tuple(representations)
        fallback = next((name for name in ("w4a16", "bf16") if name in available), None)
        selected = f"{fallback}_fallback" if fallback else "unavailable_fallback"
        if self.strict:
            raise RuntimeError(
                f"FlatQuant strict dispatch rejected prefix={prefix}, M={rows}, "
                f"selected={selected}"
            )
        if fallback is None:
            raise RuntimeError(
                f"FlatQuant fallback representation not exported for prefix={prefix}, "
                f"M={rows}, selected={selected}"
            )
        return fallback


def _whole_tensor_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    if loaded_weight.shape != param.shape:
        raise ValueError(
            f"expected shape {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
        )
    if loaded_weight.dtype != param.dtype:
        raise ValueError(f"expected dtype {param.dtype}, got {loaded_weight.dtype}")
    param.data.copy_(loaded_weight)


def _scalar_tensor_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    if loaded_weight.numel() != 1:
        raise ValueError(f"expected scalar tensor, got shape {tuple(loaded_weight.shape)}")
    if loaded_weight.dtype != param.dtype:
        raise ValueError(f"expected dtype {param.dtype}, got {loaded_weight.dtype}")
    param.data.copy_(loaded_weight.reshape(param.shape))


def _parameter(shape, dtype, attrs):
    parameter = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
    set_weight_attrs(parameter, attrs)
    return parameter


class FlatQuantW4A4LinearMethod(LinearMethodBase):
    def __init__(self, prefix: str, policy: DispatchPolicy, representations):
        self.prefix = prefix
        self.policy = policy
        self.representations = tuple(representations)

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("FlatQuant W4A4 supports TP=1 only")
        if input_size_per_partition % 2:
            raise ValueError("FlatQuant W4A4 packed weights require an even input size")

        output_size_per_partition = sum(output_partition_sizes)
        weight_attrs = dict(extra_weight_attrs)
        weight_attrs.update(
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            pack_factor=2,
            weight_loader=_whole_tensor_loader,
        )
        layer.register_parameter(
            "weight",
            _parameter(
                (output_size_per_partition, input_size_per_partition // 2),
                torch.uint8,
                weight_attrs,
            ),
        )

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs.update(output_dim=0, weight_loader=_whole_tensor_loader)
        layer.register_parameter(
            "weight_scale",
            _parameter(
                (output_size_per_partition, 1), torch.float16, scale_attrs
            ),
        )

        unsharded_attrs = dict(extra_weight_attrs)
        unsharded_attrs.update(ignore_warning=True, weight_loader=_whole_tensor_loader)
        clip_attrs = dict(unsharded_attrs)
        clip_attrs["weight_loader"] = _scalar_tensor_loader
        layer.register_parameter(
            "activation_clip", _parameter((1,), torch.float16, clip_attrs)
        )

        if self.prefix.endswith(".o_proj"):
            left_size, right_size = 40, None
        else:
            left_size, right_size = decompose_dim(input_size)
        layer.register_parameter(
            "flatquant_left",
            _parameter((left_size, left_size), params_dtype, unsharded_attrs),
        )
        if right_size is not None:
            layer.register_parameter(
                "flatquant_right",
                _parameter((right_size, right_size), params_dtype, unsharded_attrs),
            )

    def apply(self, layer, x, bias=None):
        return apply_w4a4(
            layer, x, bias, prefix=self.prefix, policy=self.policy,
            representations=self.representations
        )


@register_quantization_config("flatquant_w4a4")
class FlatQuantW4A4Config(QuantizationConfig):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.policy = DispatchPolicy.from_env()
        self.representations = tuple(config.get("representations", ("w4a4",)))
        unsupported = set(self.representations) - {"w4a4"}
        if unsupported:
            raise ValueError(
                "fallback representations are declared but this exporter/runtime has no "
                f"matching tensor load path: {sorted(unsupported)}"
            )

    @classmethod
    def get_config_filenames(cls):
        return ["flatquant_w4a4_config.json"]

    @classmethod
    def from_config(cls, config):
        validate_manifest(config)
        return cls(config)

    def get_name(self):
        return "flatquant_w4a4"

    @classmethod
    def get_min_capability(cls):
        return 80

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    def get_quant_method(self, layer, prefix):
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("FlatQuant W4A4 supports TP=1 only")
        suffixes = (".qkv_proj", ".o_proj", ".gate_up_proj", ".down_proj")
        if "language_model.model.layers." in prefix and prefix.endswith(suffixes):
            return FlatQuantW4A4LinearMethod(
                prefix, self.policy, self.representations
            )
        # Embedding classes also consult this hook and must select their own
        # UnquantizedEmbeddingMethod when the quant config does not target them.
        if hasattr(layer, "num_embeddings") and hasattr(layer, "embedding_dim"):
            return None
        return UnquantizedLinearMethod()

    def get_embedding_quant_method(self, layer, prefix):
        return None

    def get_attention_quant_method(self, layer, prefix):
        return None
