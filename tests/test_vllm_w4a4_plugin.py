import pytest
import torch
from torch import nn

from flatquant_w4a4.format import validate_manifest
from flatquant_vllm_plugin import w4a4_config
from flatquant_vllm_plugin.w4a4_config import (
    FlatQuantW4A4Config,
    FlatQuantW4A4LinearMethod,
)
from flatquant_vllm_plugin.w4a4_ops import DispatchCounters


class FakeLinear(nn.Module):
    pass


@pytest.fixture
def manifest():
    value = {
        "format_version": 1,
        "quant_method": "flatquant_w4a4",
        "w_bits": 4,
        "a_bits": 4,
        "packing": "signed-low-nibble-first",
        "target_device": "sm80",
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "targets": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    }
    validate_manifest(value)
    return value


@pytest.fixture
def config(manifest, monkeypatch):
    monkeypatch.setattr(w4a4_config, "get_tensor_model_parallel_world_size", lambda: 1)
    return FlatQuantW4A4Config.from_config(manifest)


def test_config_contract(config):
    assert config.get_name() == "flatquant_w4a4"
    assert config.get_config_filenames() == ["flatquant_w4a4_config.json"]
    assert config.get_min_capability() == 80
    assert config.get_supported_act_dtypes() == [torch.bfloat16]


def test_only_exaone_text_projections_are_wrapped(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    assert isinstance(method, FlatQuantW4A4LinearMethod)
    assert config.get_quant_method(FakeLinear(), "visual.blocks.0.mlp.down_proj") is None


def test_tp_two_is_rejected(monkeypatch, config):
    monkeypatch.setattr(w4a4_config, "get_tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="TP=1"):
        config.get_quant_method(
            FakeLinear(), "language_model.model.layers.0.mlp.down_proj"
        )


def _create(method, input_size=64, output_parts=(5, 7)):
    layer = FakeLinear()
    method.create_weights(
        layer,
        input_size_per_partition=input_size,
        output_partition_sizes=list(output_parts),
        input_size=input_size,
        output_size=sum(output_parts),
        params_dtype=torch.bfloat16,
    )
    return layer


def test_create_weights_registers_native_packed_shapes(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method)

    assert layer.weight.shape == (12, 32)
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (12, 1)
    assert layer.activation_clip.shape == (1,)
    assert isinstance(layer.activation_clip, nn.Parameter)
    assert layer.flatquant_left.shape == (8, 8)
    assert layer.flatquant_right.shape == (8, 8)
    assert layer.weight.output_dim == 0
    assert layer.weight.input_dim == 1
    assert layer.weight.packed_dim == 1
    assert layer.weight.pack_factor == 2
    assert layer.weight_scale.output_dim == 0


def test_o_proj_registers_left_only_transform(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.o_proj"
    )
    layer = _create(method, input_size=1600, output_parts=(10,))
    assert layer.flatquant_left.shape == (40, 40)
    assert not hasattr(layer, "flatquant_right")


def test_native_loaders_copy_fused_rows_without_repacking(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method)
    packed = torch.arange(12 * 32, dtype=torch.int64).to(torch.uint8).reshape(12, 32)
    scales = torch.arange(12, dtype=torch.float32).reshape(12, 1)

    layer.weight.weight_loader(layer.weight, packed)
    layer.weight_scale.weight_loader(layer.weight_scale, scales)

    assert torch.equal(layer.weight, packed)
    assert torch.equal(layer.weight_scale, scales)
    assert torch.equal(layer.weight[5], packed[5])


def test_native_loader_rejects_partial_or_transposed_tensor(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method)
    with pytest.raises(ValueError, match="expected shape"):
        layer.weight.weight_loader(layer.weight, torch.empty(32, 12, dtype=torch.uint8))


def test_activation_clip_loader_accepts_exported_scalar(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method)
    layer.activation_clip.weight_loader(layer.activation_clip, torch.tensor(0.75))
    assert layer.activation_clip.shape == (1,)
    assert layer.activation_clip.item() == pytest.approx(0.75, abs=1e-3)


def test_dispatch_counters_snapshot_is_a_copy():
    counters = DispatchCounters()
    counters.increment("w4a4")
    snapshot = counters.snapshot()
    snapshot["w4a4"] = 99
    assert counters.snapshot() == {"w4a4": 1}


def test_import_registers_native_custom_operators():
    assert hasattr(torch.ops.flatquant, "quantize_pack_i4")
    assert hasattr(torch.ops.flatquant, "w4a4_linear")
