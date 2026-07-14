import pytest
import torch

from flatquant_w4a4.format import validate_manifest
from flatquant_w4a4.packing import merge_output_rows, pack_signed_i4, unpack_signed_i4


def valid_manifest():
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


def test_signed_i4_round_trip_including_boundaries():
    values = torch.tensor([[-8, -7, -1, 0, 1, 6, 7, -8]], dtype=torch.int8)
    packed = pack_signed_i4(values)
    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 4)
    torch.testing.assert_close(unpack_signed_i4(packed), values, rtol=0, atol=0)


def test_signed_i4_packing_uses_low_nibble_first():
    values = torch.tensor([[1, 2, -8, -1]], dtype=torch.int8)
    expected = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)
    torch.testing.assert_close(pack_signed_i4(values), expected, rtol=0, atol=0)


def test_signed_i4_unpacking_decodes_low_nibble_first():
    packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)
    expected = torch.tensor([[1, 2, -8, -1]], dtype=torch.int8)
    torch.testing.assert_close(unpack_signed_i4(packed), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_signed_i4_unpacking_preserves_cuda_device():
    packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8, device="cuda")
    unpacked = unpack_signed_i4(packed)
    assert unpacked.device == packed.device


def test_merge_fused_projection_preserves_output_rows():
    q = torch.tensor([[-8, 7], [1, 2]], dtype=torch.int8)
    k = torch.tensor([[3, 4]], dtype=torch.int8)
    merged = merge_output_rows([pack_signed_i4(q), pack_signed_i4(k)])
    torch.testing.assert_close(unpack_signed_i4(merged), torch.cat([q, k]), rtol=0, atol=0)


@pytest.mark.parametrize("key,value", [("w_bits", 8), ("a_bits", 8), ("target_device", "sm90"), ("tensor_parallel_size", 2)])
def test_manifest_rejects_unsupported_contract(key, value):
    manifest = valid_manifest()
    manifest[key] = value
    with pytest.raises(ValueError, match=key):
        validate_manifest(manifest)
