from typing import Any

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.linear import LinearMethodBase

from .transform import apply_transform, decompose_dim


class FlatQuantLinearMethod(LinearMethodBase):
    def __init__(self, inner, prefix):
        self.inner = inner
        self.prefix = prefix

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
            raise NotImplementedError("FlatQuant vLLM currently supports TP=1 only.")
        self.inner.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        if self.prefix.endswith(".o_proj"):
            left_size = 40
            right_size = None
        else:
            left_size, right_size = decompose_dim(input_size)
        layer.register_parameter(
            "flatquant_left",
            nn.Parameter(torch.empty(left_size, left_size, dtype=params_dtype), requires_grad=False),
        )
        if right_size is not None:
            layer.register_parameter(
                "flatquant_right",
                nn.Parameter(torch.empty(right_size, right_size, dtype=params_dtype), requires_grad=False),
            )

    def process_weights_after_loading(self, layer):
        self.inner.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.inner.apply(layer, apply_transform(layer, x), bias=bias)


@register_quantization_config("flatquant")
class FlatQuantConfig(CompressedTensorsConfig):
    @classmethod
    def get_config_filenames(cls):
        return ["flatquant_vllm_config.json"]

    def get_name(self):
        return "flatquant"

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return super().from_config(config)

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        transform_suffixes = (".qkv_proj", ".gate_up_proj", ".down_proj", ".o_proj")
        # Only the language-model decoder layers carry FlatQuant transforms.
        # The vision tower exposes ``visual.blocks.N.mlp.down_proj``, which also
        # ends with ``.down_proj`` but is in the compressed-tensors ignore list
        # (so ``super()`` returns an UnquantizedLinearMethod, not None) and has
        # no exported ``flatquant_left/right`` weights. Wrapping it would apply an
        # uninitialized Kronecker transform to the vision activations and corrupt
        # image embeddings, leaving text/ppl intact while multimodal prefill
        # degenerates. Restrict wrapping to the language-model decoder.
        is_language_model = "language_model" in prefix and "visual" not in prefix
        if (
            method is not None
            and is_language_model
            and prefix.endswith(transform_suffixes)
        ):
            return FlatQuantLinearMethod(method, prefix)
        return method
