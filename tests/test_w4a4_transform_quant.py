import pytest
import torch
import deploy._CUDA  # noqa: F401

from flatquant_vllm_plugin.transform import apply_transform
from flatquant_vllm_plugin.w4a4_transform_quant import transform_quantize_pack


class _Layer:
    pass


def _factors(kind, device):
    torch.manual_seed(11)
    if kind == "left":
        left = torch.randn(40, 40, device=device, dtype=torch.bfloat16) / 40
        return left, None, 6400
    left_size, right_size = ((64, 80) if kind == "kron64" else (128, 214))
    left = torch.randn(left_size, left_size, device=device, dtype=torch.bfloat16) / left_size
    right = torch.randn(right_size, right_size, device=device, dtype=torch.bfloat16) / right_size
    return left, right, left_size * right_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 16, 128, 1024])
@pytest.mark.parametrize("kind", ["left", "kron64", "kron128"])
@pytest.mark.parametrize("clip_value", [0.625])
def test_fused_matches_composed_bytes_and_scales(rows, kind, clip_value):
    left, right, width = _factors(kind, "cuda")
    source = torch.randn(width, rows, device="cuda", dtype=torch.bfloat16).T
    assert not source.is_contiguous() or rows == 1
    x = source.contiguous()
    # Pin values at and beyond the clipping boundary.
    x[0, :4] = torch.tensor([-20, -1, 1, 20], device="cuda", dtype=x.dtype)
    clip = torch.tensor([clip_value], device="cuda", dtype=torch.float16)

    layer = _Layer()
    layer.flatquant_left = left
    if right is not None:
        layer.flatquant_right = right
    transformed = apply_transform(layer, x)
    expected_packed, expected_scale = torch.ops.flatquant.quantize_pack_i4(
        transformed.contiguous(), clip
    )

    packed, scale = transform_quantize_pack(x, left, right, clip)

    assert packed.shape == (rows, width // 2)
    assert packed.dtype == torch.uint8
    assert scale.shape == (rows, 1)
    assert scale.dtype == torch.float32
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_api_accepts_noncontiguous_logical_input_at_boundary():
    left, right, width = _factors("kron64", "cuda")
    x = torch.randn(width, 16, device="cuda", dtype=torch.bfloat16).T
    assert not x.is_contiguous()
    clip = torch.ones(1, device="cuda", dtype=torch.float16)
    packed, scales = transform_quantize_pack(x, left, right, clip)
    expected = transform_quantize_pack(x.contiguous(), left, right, clip)
    torch.testing.assert_close(packed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(scales, expected[1], rtol=0, atol=0)
