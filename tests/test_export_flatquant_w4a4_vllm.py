import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from flatquant_w4a4.format import validate_manifest
from flatquant_w4a4.packing import pack_signed_i4
from tools.export_flatquant_w4a4_vllm import export_checkpoint


PREFIX = "language_model.model.layers.0."
Q_ROWS, K_ROWS, V_ROWS = 4, 2, 2
K = 64


def _weight(rows):
    values = (torch.arange(rows * K, dtype=torch.int64) % 16 - 8).to(torch.int8)
    return pack_signed_i4(values.reshape(rows, K))


def make_one_layer_source_checkpoint(path: Path) -> Path:
    path.mkdir()
    projections = {
        "self_attn.q_proj": Q_ROWS,
        "self_attn.k_proj": K_ROWS,
        "self_attn.v_proj": V_ROWS,
        "self_attn.o_proj": 4,
        "mlp.gate_proj": 6,
        "mlp.up_proj": 6,
        "mlp.down_proj": 4,
    }
    tensors = {}
    for index, (name, rows) in enumerate(projections.items()):
        tensors[PREFIX + name + ".weight"] = _weight(rows)
        tensors[PREFIX + name + ".weight_scale"] = torch.full((rows, 1), index + 1.0)

    tensors.update(
        {
            PREFIX + "self_attn.qkv_trans.matrix_left": torch.eye(8),
            PREFIX + "self_attn.qkv_trans.matrix_right": torch.eye(8),
            PREFIX + "self_attn.o_trans.matrix": torch.eye(8),
            PREFIX + "mlp.up_gate_trans.matrix_left": torch.eye(8),
            PREFIX + "mlp.up_gate_trans.matrix_right": torch.eye(8),
            PREFIX + "mlp.down_trans.matrix_left": torch.eye(8),
            PREFIX + "mlp.down_trans.matrix_right": torch.eye(8),
            PREFIX + "self_attn.q_proj.activation_clip": torch.tensor(0.9),
            PREFIX + "self_attn.o_proj.activation_clip": torch.tensor(0.8),
            PREFIX + "mlp.gate_proj.activation_clip": torch.tensor(0.7),
            PREFIX + "mlp.down_proj.activation_clip": torch.tensor(0.6),
        }
    )
    keys = sorted(tensors)
    shards = [
        dict((key, tensors[key]) for key in keys[::2]),
        dict((key, tensors[key]) for key in keys[1::2]),
    ]
    weight_map = {}
    for number, shard in enumerate(shards, 1):
        name = f"model-{number:05d}-of-00002.safetensors"
        save_file(shard, path / name)
        weight_map.update(dict.fromkeys(shard, name))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (path / "config.json").write_text(json.dumps({"architectures": ["Exaone4ForCausalLM"]}))
    (path / "tokenizer_config.json").write_text("{}")
    return path


def test_export_fuses_rows_and_writes_native_manifest(tmp_path):
    source = make_one_layer_source_checkpoint(tmp_path / "source")
    output = tmp_path / "output"
    export_checkpoint(source, output)

    manifest = json.loads((output / "flatquant_w4a4_config.json").read_text())
    validate_manifest(manifest)
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == manifest
    assert (output / "tokenizer_config.json").exists()
    with safe_open(output / "model-00001-of-00001.safetensors", framework="pt") as handle:
        qkv = handle.get_tensor(PREFIX + "self_attn.qkv_proj.weight")
        scales = handle.get_tensor(PREFIX + "self_attn.qkv_proj.weight_scale")
        assert qkv.dtype == torch.uint8
        assert qkv.shape == (Q_ROWS + K_ROWS + V_ROWS, K // 2)
        assert scales.shape == (Q_ROWS + K_ROWS + V_ROWS, 1)
        assert PREFIX + "self_attn.qkv_proj.flatquant_left" in handle.keys()
        assert PREFIX + "self_attn.qkv_proj.activation_clip" in handle.keys()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {"model-00001-of-00001.safetensors"}


def test_missing_activation_clip_names_offending_tensor(tmp_path):
    source = make_one_layer_source_checkpoint(tmp_path / "source")
    _remove_tensor(source, PREFIX + "self_attn.q_proj.activation_clip")
    with pytest.raises(ValueError, match="self_attn.q_proj.activation_clip"):
        export_checkpoint(source, tmp_path / "output")


def test_non_exaone_architecture_names_config_field(tmp_path):
    source = make_one_layer_source_checkpoint(tmp_path / "source")
    (source / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"]})
    )
    with pytest.raises(ValueError, match="architectures"):
        export_checkpoint(source, tmp_path / "output")


def test_unaligned_packed_k_names_offending_tensor(tmp_path):
    source = make_one_layer_source_checkpoint(tmp_path / "source")
    name = PREFIX + "self_attn.q_proj.weight"
    _replace_tensor(source, name, torch.zeros((Q_ROWS, 31), dtype=torch.uint8))
    with pytest.raises(ValueError, match="self_attn.q_proj.weight"):
        export_checkpoint(source, tmp_path / "output")


def test_documented_cli_is_directly_executable():
    script = Path(__file__).parents[1] / "tools" / "export_flatquant_w4a4_vllm.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--source" in result.stdout


def _replace_tensor(source, name, replacement=None):
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shard = source / index["weight_map"][name]
    with safe_open(shard, framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys() if key != name}
    if replacement is not None:
        tensors[name] = replacement
    save_file(tensors, shard)
    if replacement is None:
        del index["weight_map"][name]
        (source / "model.safetensors.index.json").write_text(json.dumps(index))


def _remove_tensor(source, name):
    _replace_tensor(source, name)
