import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from flatquant_w4a4.format import validate_manifest
from flatquant_vllm_plugin import w4a4_config
from flatquant_vllm_plugin.w4a4_config import (
    DispatchPolicy,
    FlatQuantW4A4Config,
    FlatQuantW4A4LinearMethod,
)
from flatquant_vllm_plugin.w4a4_ops import (
    DispatchCounters,
    apply_w4a4,
    dispatch_counters,
    selected_w4a4_projections,
)


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
        "representations": ["w4a4"],
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
    selected_w4a4_projections.reset()
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    assert isinstance(method, FlatQuantW4A4LinearMethod)
    assert selected_w4a4_projections.snapshot() == (
        "language_model.model.layers.0.self_attn.qkv_proj",
    )
    assert isinstance(
        config.get_quant_method(FakeLinear(), "visual.blocks.0.mlp.down_proj"),
        UnquantizedLinearMethod,
    )


def test_selection_accounting_deduplicates_projection_prefixes(config):
    selected_w4a4_projections.reset()
    prefix = "language_model.model.layers.0.mlp.down_proj"
    config.get_quant_method(FakeLinear(), prefix)
    config.get_quant_method(FakeLinear(), prefix)
    assert selected_w4a4_projections.snapshot() == (prefix,)


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
    scales = torch.arange(12, dtype=torch.float16).reshape(12, 1)

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
    layer.activation_clip.weight_loader(
        layer.activation_clip, torch.tensor(0.75, dtype=torch.float16)
    )
    assert layer.activation_clip.shape == (1,)
    assert layer.activation_clip.item() == pytest.approx(0.75, abs=1e-3)


def test_dispatch_counters_snapshot_is_a_copy():
    counters = DispatchCounters()
    counters.increment("w4a4")
    snapshot = counters.snapshot()
    snapshot["w4a4"] = 99
    assert counters.snapshot() == {
        "w4a4": 1, "w4a16_fallback": 0, "bf16_fallback": 0
    }


def test_dispatch_counters_can_be_reset_for_worker_rpc_observability():
    counters = DispatchCounters()
    counters.increment("w4a4")
    assert counters.reset() == {
        "w4a4": 1, "w4a16_fallback": 0, "bf16_fallback": 0
    }
    assert counters.snapshot() == {
        "w4a4": 0, "w4a16_fallback": 0, "bf16_fallback": 0
    }


def test_dispatch_policy_environment_defaults_to_w4a4_only(monkeypatch):
    monkeypatch.delenv("FLATQUANT_W4A4_MIN_ROWS", raising=False)
    monkeypatch.delenv("FLATQUANT_W4A4_STRICT", raising=False)
    assert DispatchPolicy.from_env() == DispatchPolicy(min_w4a4_rows=1, strict=False)


def test_dispatch_policy_parses_threshold_and_strict(monkeypatch):
    monkeypatch.setenv("FLATQUANT_W4A4_MIN_ROWS", "8")
    monkeypatch.setenv("FLATQUANT_W4A4_STRICT", "1")
    assert DispatchPolicy.from_env() == DispatchPolicy(min_w4a4_rows=8, strict=True)


@pytest.mark.parametrize("value", ["true", "2", "", "yes"])
def test_dispatch_policy_rejects_invalid_strict(monkeypatch, value):
    monkeypatch.setenv("FLATQUANT_W4A4_STRICT", value)
    with pytest.raises(ValueError, match="FLATQUANT_W4A4_STRICT"):
        DispatchPolicy.from_env()


@pytest.mark.parametrize("value", ["not-an-int", "0", "-2"])
def test_dispatch_policy_rejects_invalid_min_rows(monkeypatch, value):
    monkeypatch.setenv("FLATQUANT_W4A4_MIN_ROWS", value)
    with pytest.raises(ValueError, match="FLATQUANT_W4A4_MIN_ROWS"):
        DispatchPolicy.from_env()


def test_dispatch_policy_selects_w4a4_at_threshold():
    policy = DispatchPolicy(min_w4a4_rows=8, strict=False)
    assert policy.select(8, "layer.0.qkv_proj", ("w4a4", "w4a16")) == "w4a4"


def test_dispatch_policy_rejects_unexported_requested_fallback():
    policy = DispatchPolicy(min_w4a4_rows=8, strict=False)
    with pytest.raises(RuntimeError, match="fallback representation.*not exported"):
        policy.select(2, "layer.0.qkv_proj", ("w4a4",))


def test_dispatch_policy_strict_error_identifies_prefix_m_and_selection():
    policy = DispatchPolicy(min_w4a4_rows=8, strict=True)
    with pytest.raises(
        RuntimeError, match=r"layer\.0\.down_proj.*M=2.*w4a16_fallback"
    ):
        policy.select(2, "layer.0.down_proj", ("w4a4", "w4a16"))


def test_dispatch_counter_rpc_schema_distinguishes_all_paths():
    counters = DispatchCounters()
    counters.increment("w4a16_fallback")
    counters.increment("bf16_fallback")
    assert counters.snapshot() == {
        "w4a4": 0,
        "w4a16_fallback": 1,
        "bf16_fallback": 1,
    }


def test_import_registers_native_custom_operators():
    assert hasattr(torch.ops.flatquant, "quantize_pack_i4")
    assert hasattr(torch.ops.flatquant, "w4a4_linear")


@pytest.mark.parametrize(
    ("parameter_name", "wrong_dtype"),
    [
        ("weight", torch.int8),
        ("weight_scale", torch.float32),
        ("flatquant_left", torch.float16),
        ("flatquant_right", torch.float16),
        ("activation_clip", torch.float32),
    ],
)
def test_loaders_reject_wrong_dtype_for_every_parameter_family(
    config, parameter_name, wrong_dtype
):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method)
    parameter = getattr(layer, parameter_name)
    loaded = torch.empty(parameter.shape, dtype=wrong_dtype)
    with pytest.raises(ValueError, match="expected dtype"):
        parameter.weight_loader(parameter, loaded)


def test_all_native_parameters_have_direct_whole_tensor_loaders(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method)
    names = (
        "weight",
        "weight_scale",
        "flatquant_left",
        "flatquant_right",
        "activation_clip",
    )
    assert all(callable(getattr(layer, name).weight_loader) for name in names)

    expected = {
        "weight": torch.arange(layer.weight.numel(), dtype=torch.int64)
        .to(torch.uint8)
        .reshape(layer.weight.shape),
        "weight_scale": torch.arange(
            layer.weight_scale.numel(), dtype=torch.float16
        ).reshape(layer.weight_scale.shape),
        "flatquant_left": torch.arange(
            layer.flatquant_left.numel(), dtype=torch.bfloat16
        ).reshape(layer.flatquant_left.shape),
        "flatquant_right": torch.arange(
            layer.flatquant_right.numel(), dtype=torch.bfloat16
        ).reshape(layer.flatquant_right.shape),
        "activation_clip": torch.tensor(0.625, dtype=torch.float16),
    }
    for name, loaded in expected.items():
        parameter = getattr(layer, name)
        parameter.weight_loader(parameter, loaded)
        assert torch.equal(parameter, loaded.reshape(parameter.shape))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_apply_w4a4_cuda_preserves_leading_dims_bias_and_counts(config):
    method = config.get_quant_method(
        FakeLinear(), "language_model.model.layers.0.self_attn.qkv_proj"
    )
    layer = _create(method, input_size=256, output_parts=(8,)).cuda()
    with torch.no_grad():
        layer.weight.fill_(0x11)
        layer.weight_scale.fill_(0.25)
        layer.activation_clip.fill_(1)
        layer.flatquant_left.copy_(
            torch.eye(16, dtype=torch.bfloat16, device="cuda")
        )
        layer.flatquant_right.copy_(
            torch.eye(16, dtype=torch.bfloat16, device="cuda")
        )

    x = torch.linspace(-1, 1, 2 * 3 * 256, device="cuda", dtype=torch.bfloat16)
    x = x.reshape(2, 3, 256)
    bias = torch.arange(8, device="cuda", dtype=torch.bfloat16)
    before = dispatch_counters.snapshot().get("w4a4", 0)

    packed_x, x_scale = torch.ops.flatquant.quantize_pack_i4(
        x.reshape(-1, 256).contiguous(), layer.activation_clip
    )
    expected = torch.ops.flatquant.w4a4_linear(
        packed_x, layer.weight, x_scale, layer.weight_scale, torch.bfloat16
    ).reshape(2, 3, 8)

    without_bias = apply_w4a4(layer, x)
    with_bias = method.apply(layer, x, bias)

    assert without_bias.shape == (2, 3, 8)
    assert without_bias.dtype == torch.bfloat16
    assert torch.count_nonzero(without_bias) > 0
    torch.testing.assert_close(without_bias, expected)
    torch.testing.assert_close(with_bias, expected + bias)
    assert dispatch_counters.snapshot()["w4a4"] == before + 2
