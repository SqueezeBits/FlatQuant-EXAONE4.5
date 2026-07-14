import json

import pytest

from tools.export_flatquant_vllm import (
    TransformSelection,
    parse_transform_selection,
    validate_transform_selection,
)


def test_empty_transform_selection_preserves_current_export():
    selection = parse_transform_selection([])

    assert selection == TransformSelection()
    assert selection.to_manifest(num_hidden_layers=3)["excluded"] == []
    assert len(selection.to_manifest(num_hidden_layers=3)["included"]) == 12


def test_parse_projection_families_and_inclusive_layer_ranges():
    selection = parse_transform_selection(
        ["qkv_proj:0-3", "o_proj:7", "gate_up_proj:10-12", "down_proj"]
    )

    assert selection.excludes("qkv_proj", 0)
    assert selection.excludes("qkv_proj", 3)
    assert not selection.excludes("qkv_proj", 4)
    assert selection.excludes("o_proj", 7)
    assert not selection.excludes("o_proj", 8)
    assert selection.excludes("gate_up_proj", 11)
    assert selection.excludes("down_proj", 999)


@pytest.mark.parametrize(
    "spec",
    ["q_proj", "qkv_proj:", "qkv_proj:3-1", "o_proj:-2", "down_proj:one"],
)
def test_invalid_transform_selection_is_rejected(spec):
    with pytest.raises(ValueError):
        parse_transform_selection([spec])


def test_manifest_lists_every_included_and_excluded_projection():
    selection = parse_transform_selection(["qkv_proj:1", "down_proj:0-1"])

    manifest = selection.to_manifest(num_hidden_layers=2)

    assert manifest["format_version"] == 1
    assert manifest["excluded"] == [
        {"layer": 0, "projection": "down_proj"},
        {"layer": 1, "projection": "qkv_proj"},
        {"layer": 1, "projection": "down_proj"},
    ]
    assert len(manifest["included"]) == 5


def test_packed_checkpoint_cannot_be_relabelled_as_selective(tmp_path):
    selection = parse_transform_selection(["qkv_proj:0-3"])

    with pytest.raises(ValueError, match="recalibrated and requantized"):
        validate_transform_selection(tmp_path, selection)


def test_nonempty_selection_is_rejected_even_with_untrusted_metadata(tmp_path):
    (tmp_path / "flatquant_transform_selection.json").write_text(
        json.dumps({"excluded": [{"layer": 0, "projection": "qkv_proj"}]})
    )

    with pytest.raises(ValueError, match="packed checkpoint"):
        validate_transform_selection(
            tmp_path, parse_transform_selection(["qkv_proj:0"])
        )
