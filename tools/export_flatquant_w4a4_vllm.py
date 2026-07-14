"""Export native packed FlatQuant W4A4 weights for the EXAONE vLLM plugin."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flatquant_w4a4.packing import merge_output_rows


FUSED_PROJECTIONS = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}
UNFUSED_PROJECTIONS = {
    "self_attn.o_proj": "self_attn.o_proj",
    "mlp.down_proj": "mlp.down_proj",
}
TRANSFORM_RENAMES = {
    "self_attn.qkv_trans.matrix_left": "self_attn.qkv_proj.flatquant_left",
    "self_attn.qkv_trans.matrix_right": "self_attn.qkv_proj.flatquant_right",
    "self_attn.o_trans.matrix": "self_attn.o_proj.flatquant_left",
    "mlp.up_gate_trans.matrix_left": "mlp.gate_up_proj.flatquant_left",
    "mlp.up_gate_trans.matrix_right": "mlp.gate_up_proj.flatquant_right",
    "mlp.down_trans.matrix_left": "mlp.down_proj.flatquant_left",
    "mlp.down_trans.matrix_right": "mlp.down_proj.flatquant_right",
}
CLIP_RENAMES = {
    "self_attn.q_proj.activation_clip": "self_attn.qkv_proj.activation_clip",
    "self_attn.o_proj.activation_clip": "self_attn.o_proj.activation_clip",
    "mlp.gate_proj.activation_clip": "mlp.gate_up_proj.activation_clip",
    "mlp.down_proj.activation_clip": "mlp.down_proj.activation_clip",
}
SM80_K_ALIGNMENT = 64


def build_manifest() -> dict:
    return {
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


def _layer_prefixes(tensors):
    marker = "self_attn.q_proj.weight"
    return sorted(key[: -len(marker)] for key in tensors if key.endswith(marker))


def _required(tensors, name):
    try:
        return tensors[name]
    except KeyError as error:
        raise ValueError(f"Missing required tensor {name}") from error


def _validate_weight(name, weight):
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError(f"{name} must be a 2D uint8 packed tensor")
    logical_k = weight.shape[1] * 2
    if logical_k % SM80_K_ALIGNMENT:
        raise ValueError(
            f"{name} has logical K={logical_k}, not divisible by SM80 alignment {SM80_K_ALIGNMENT}"
        )


def fuse_projection_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(tensors)
    prefixes = _layer_prefixes(tensors)
    if not prefixes:
        raise ValueError("Missing required tensor self_attn.q_proj.weight")

    for prefix in prefixes:
        for target, sources in FUSED_PROJECTIONS.items():
            weight_names = [prefix + source + ".weight" for source in sources]
            scale_names = [prefix + source + ".weight_scale" for source in sources]
            weights = [_required(tensors, name) for name in weight_names]
            scales = [_required(tensors, name) for name in scale_names]
            for name, weight in zip(weight_names, weights):
                _validate_weight(name, weight)
            for name, weight, scale in zip(scale_names, weights, scales):
                if scale.ndim != 2 or scale.shape != (weight.shape[0], 1):
                    raise ValueError(f"{name} must have shape ({weight.shape[0]}, 1)")
            output[prefix + target + ".weight"] = merge_output_rows(weights)
            output[prefix + target + ".weight_scale"] = torch.cat(scales, dim=0).contiguous()
            for name in weight_names + scale_names:
                del output[name]

        for target, source in UNFUSED_PROJECTIONS.items():
            weight_name = prefix + source + ".weight"
            scale_name = prefix + source + ".weight_scale"
            weight = _required(tensors, weight_name)
            _validate_weight(weight_name, weight)
            scale = _required(tensors, scale_name)
            if scale.ndim != 2 or scale.shape != (weight.shape[0], 1):
                raise ValueError(f"{scale_name} must have shape ({weight.shape[0]}, 1)")
            if source != target:
                output[prefix + target + ".weight"] = output.pop(weight_name)
                output[prefix + target + ".weight_scale"] = output.pop(scale_name)

        for source, target in TRANSFORM_RENAMES.items():
            source_name = prefix + source
            output[prefix + target] = (
                output.pop(source_name)
                if source_name in output
                else _required(tensors, source_name)
            )
        for source, target in CLIP_RENAMES.items():
            source_name = prefix + source
            output[prefix + target] = (
                output.pop(source_name)
                if source_name in output
                else _required(tensors, source_name)
            )
    return output


def _read_indexed_tensors(source: Path) -> dict[str, torch.Tensor]:
    index_path = source / "model.safetensors.index.json"
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors index {index_path}") from error
    tensors = {}
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                if weight_map.get(key) != shard_name:
                    raise ValueError(f"Safetensors index does not map tensor {key}")
                tensors[key] = handle.get_tensor(key)
    for key in weight_map:
        if key not in tensors:
            raise ValueError(f"Safetensors index references missing tensor {key}")
    return tensors


def export_checkpoint(source: Path, output: Path) -> None:
    source, output = Path(source), Path(output)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    architectures = config.get("architectures", [])
    if not architectures or not all("exaone" in name.lower() for name in architectures):
        raise ValueError(f"config.json architectures is not EXAONE: {architectures!r}")

    converted = fuse_projection_tensors(_read_indexed_tensors(source))
    output.mkdir(parents=True, exist_ok=True)
    shard_name = "model-00001-of-00001.safetensors"
    ordered = {key: converted[key].contiguous() for key in sorted(converted)}
    save_file(ordered, output / shard_name, metadata={"format": "pt"})
    weight_map = dict.fromkeys(ordered, shard_name)
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in ordered.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    manifest = build_manifest()
    (output / "flatquant_w4a4_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    config["quantization_config"] = manifest
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    generated = {
        "config.json",
        "model.safetensors.index.json",
        shard_name,
        "flatquant_w4a4_config.json",
    }
    for path in sorted(source.iterdir()):
        if (
            path.is_file()
            and path.name not in generated
            and path.suffix not in {".safetensors", ".pth"}
        ):
            shutil.copy2(path, output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()
