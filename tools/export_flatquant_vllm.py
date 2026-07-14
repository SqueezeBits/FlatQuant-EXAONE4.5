"""Convert a packed FlatQuant W4A16 checkpoint to vLLM WNA16 storage."""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file


TRANSFORM_RENAMES = {
    ".self_attn.qkv_trans.matrix_left": ".self_attn.qkv_proj.flatquant_left",
    ".self_attn.qkv_trans.matrix_right": ".self_attn.qkv_proj.flatquant_right",
    ".self_attn.o_trans.matrix": ".self_attn.o_proj.flatquant_left",
    ".mlp.up_gate_trans.matrix_left": ".mlp.gate_up_proj.flatquant_left",
    ".mlp.up_gate_trans.matrix_right": ".mlp.gate_up_proj.flatquant_right",
    ".mlp.down_trans.matrix_left": ".mlp.down_proj.flatquant_left",
    ".mlp.down_trans.matrix_right": ".mlp.down_proj.flatquant_right",
}

PROJECTION_FAMILIES = (
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TransformSelection:
    """Transform exclusions requested for a freshly quantized checkpoint.

    Each rule is ``(projection, first_layer, last_layer)``. ``None`` bounds mean
    every layer. This converter intentionally cannot apply non-empty rules to an
    already packed FlatQuant checkpoint; see :func:`validate_transform_selection`.
    """

    rules: tuple[tuple[str, int | None, int | None], ...] = ()

    def excludes(self, projection, layer):
        return any(
            candidate == projection
            and (first is None or first <= layer <= last)
            for candidate, first, last in self.rules
        )

    def to_manifest(self, num_hidden_layers):
        included = []
        excluded = []
        for layer in range(num_hidden_layers):
            for projection in PROJECTION_FAMILIES:
                entry = {"layer": layer, "projection": projection}
                target = excluded if self.excludes(projection, layer) else included
                target.append(entry)
        return {
            "format_version": 1,
            "coordinate_system": "flatquant_packed",
            "included": included,
            "excluded": excluded,
        }


def parse_transform_selection(specs):
    """Parse ``PROJECTION`` or ``PROJECTION:FIRST[-LAST]`` exclusions."""
    rules = []
    for spec in specs:
        projection, separator, layer_spec = spec.partition(":")
        if projection not in PROJECTION_FAMILIES:
            raise ValueError(
                f"Unknown transform projection {projection!r}; expected one of "
                + ", ".join(PROJECTION_FAMILIES)
            )
        if not separator:
            rules.append((projection, None, None))
            continue
        if not layer_spec:
            raise ValueError(f"Missing layer or layer range in {spec!r}")
        bounds = layer_spec.split("-", 1)
        try:
            first = int(bounds[0])
            last = int(bounds[-1])
        except ValueError as error:
            raise ValueError(f"Invalid layer range in {spec!r}") from error
        if first < 0 or last < first:
            raise ValueError(f"Invalid inclusive layer range in {spec!r}")
        rules.append((projection, first, last))
    return TransformSelection(tuple(rules))


def validate_transform_selection(source, selection):
    """Prevent relabelling weights whose FlatQuant basis is already fixed."""
    if not selection.rules:
        return
    raise ValueError(
        "A packed checkpoint cannot safely omit online FlatQuant transforms: "
        "its weights are already expressed in the transformed coordinate system. "
        "The selected layers must be recalibrated and requantized without those "
        "transforms before export."
    )


def _rename_transform(key):
    for old, new in TRANSFORM_RENAMES.items():
        if key.endswith(old):
            return key[: -len(old)] + new
    return None


def _packed_target(key):
    suffix = ".linear.weight"
    if key.endswith(suffix):
        return key[: -len(suffix)] + ".weight_packed"
    return None


def _scale_target(key):
    prefix = "quantizer."
    suffix = ".linear.scale"
    if key.startswith(prefix) and key.endswith(suffix):
        base = key[len(prefix) : -len(suffix)]
        return base + ".weight_scale"
    return None


def _unpack_flatquant_int4(packed):
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    values = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.int8
    )
    values[:, 0::2] = low
    values[:, 1::2] = high
    return values


def _should_skip(key):
    return (
        key.startswith("quantizer.")
        or ".clip_factor_w_" in key
        or any(token in key for token in (".matrix_inv", ".matrix_left_inv", ".matrix_right_inv"))
        or ".vcache_trans." in key
    )


def _ct_config(ignore):
    return {
        "quant_method": "flatquant",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 128,
                    "num_bits": 4,
                    "observer": "memoryless_minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            }
        },
        "ignore": ignore,
    }


def export_checkpoint(source, output, awq_reference, transform_selection=None):
    source = Path(source)
    output = Path(output)
    transform_selection = transform_selection or TransformSelection()
    validate_transform_selection(source, transform_selection)
    output.mkdir(parents=True, exist_ok=True)

    source_config = json.loads((source / "config.json").read_text())
    awq_config = json.loads((Path(awq_reference) / "config.json").read_text())
    ct_config = _ct_config(awq_config["quantization_config"].get("ignore", []))
    source_config["quantization_config"] = ct_config
    (output / "config.json").write_text(json.dumps(source_config, indent=2) + "\n")
    (output / "flatquant_vllm_config.json").write_text(json.dumps(ct_config, indent=2) + "\n")
    selection_manifest = transform_selection.to_manifest(
        source_config["text_config"]["num_hidden_layers"]
    )
    (output / "flatquant_transform_selection.json").write_text(
        json.dumps(selection_manifest, indent=2) + "\n"
    )
    shutil.copy2(source / "quantization_config.json", output / "flatquant_source_config.json")

    index = json.loads((source / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    new_weight_map = {}
    total_size = 0

    for shard_number, shard_name in enumerate(shard_names, 1):
        print(f"Converting shard {shard_number}/{len(shard_names)}: {shard_name}")
        converted = {}
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                packed_target = _packed_target(key)
                scale_target = _scale_target(key)
                transform_target = _rename_transform(key)

                if packed_target is not None and tensor.dtype == torch.uint8:
                    values = _unpack_flatquant_int4(tensor)
                    converted[packed_target] = pack_to_int32(values, 4, packed_dim=1)
                    converted[packed_target.replace("weight_packed", "weight_shape")] = torch.tensor(
                        values.shape, dtype=torch.int64
                    )
                elif scale_target is not None:
                    out_features = tensor.shape[0]
                    source_weight = scale_target.replace(".weight_scale", ".linear.weight")
                    packed_name = "quantizer." + source_weight
                    # FlatQuant scale is per output channel. Repeating it across
                    # G128 groups preserves the exact quantized values while using
                    # vLLM's standard W4A16 storage contract.
                    input_size = next(
                        handle.get_tensor(k).shape[1] * 2
                        for k in handle.keys()
                        if k == source_weight
                    ) if source_weight in handle.keys() else None
                    if input_size is None:
                        input_size = source_config["text_config"]["hidden_size"]
                        if ".down_proj." in scale_target:
                            input_size = source_config["text_config"]["intermediate_size"]
                    converted[scale_target] = tensor.reshape(out_features, 1).to(torch.bfloat16).repeat(
                        1, input_size // 128
                    )
                elif transform_target is not None:
                    converted[transform_target] = tensor.to(torch.bfloat16)
                elif not _should_skip(key):
                    converted[key] = tensor

        output_shard = output / shard_name
        save_file(converted, output_shard, metadata={"format": "pt"})
        for key, tensor in converted.items():
            new_weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")

    generated = {"config.json", "flatquant_vllm_config.json", "flatquant_source_config.json", "flatquant_transform_selection.json", "model.safetensors.index.json", *shard_names}
    for path in source.iterdir():
        if path.name not in generated and path.is_file() and path.suffix not in {".pth", ".safetensors"}:
            shutil.copy2(path, output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--awq_reference", required=True)
    parser.add_argument(
        "--exclude-transform",
        action="append",
        default=[],
        metavar="PROJECTION[:FIRST[-LAST]]",
        help="Requires recalibration/requantization; packed inputs are refused.",
    )
    args = parser.parse_args()
    selection = parse_transform_selection(args.exclude_transform)
    export_checkpoint(
        args.source,
        args.output,
        args.awq_reference,
        transform_selection=selection,
    )


if __name__ == "__main__":
    main()
