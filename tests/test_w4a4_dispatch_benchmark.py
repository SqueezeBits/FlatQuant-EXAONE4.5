import json

import pytest
import torch
from safetensors.torch import save_file

from benchmarks.w4a4_dispatch_benchmark import load_projection_transforms


PREFIX = "model.language_model.layers.0."
NAMES = {
    "qkv_proj": ("self_attn.qkv_proj.flatquant_left", "self_attn.qkv_proj.flatquant_right"),
    "o_proj": ("self_attn.o_proj.flatquant_left",),
    "gate_up_proj": ("mlp.gate_up_proj.flatquant_left", "mlp.gate_up_proj.flatquant_right"),
    "down_proj": ("mlp.down_proj.flatquant_left", "mlp.down_proj.flatquant_right"),
}


def make_checkpoint(path, omit=None):
    tensors = {}
    for names in NAMES.values():
        for name in names:
            full = PREFIX + name
            if full != omit:
                tensors[full] = torch.eye(2, dtype=torch.bfloat16)
    shard = "model.safetensors"
    path.mkdir()
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for name in tensors}
    }))
    return path


def test_loads_exact_exported_transform_names(tmp_path):
    loaded = load_projection_transforms(make_checkpoint(tmp_path / "model"), layer=0)
    assert set(loaded) == set(NAMES)
    assert loaded["o_proj"].flatquant_left.shape == (2, 2)
    assert not hasattr(loaded["o_proj"], "flatquant_right")
    assert loaded["down_proj"].flatquant_right.shape == (2, 2)


def test_missing_learned_transform_fails_closed(tmp_path):
    missing = PREFIX + "mlp.down_proj.flatquant_right"
    with pytest.raises(ValueError, match="down_proj.flatquant_right"):
        load_projection_transforms(
            make_checkpoint(tmp_path / "model", omit=missing), layer=0
        )
