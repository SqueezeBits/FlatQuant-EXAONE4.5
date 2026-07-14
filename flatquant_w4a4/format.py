W4A4_FORMAT_VERSION = 1
TARGET_PROJECTIONS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def validate_manifest(config: dict) -> None:
    expected = {
        "format_version": W4A4_FORMAT_VERSION,
        "quant_method": "flatquant_w4a4",
        "w_bits": 4,
        "a_bits": 4,
        "packing": "signed-low-nibble-first",
        "target_device": "sm80",
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "targets": list(TARGET_PROJECTIONS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid {key}: expected {value!r}, got {config.get(key)!r}")
