import pytest
import torch

from flatquant_w4a4.packing import pack_signed_i4
from deploy.nn import LinearW4A4, quantize_pack_i4, w4a4_linear


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@CUDA
@pytest.mark.parametrize("rows", [1, 16, 128, 1024])
def test_w4a4_matches_integer_reference(rows):
    torch.manual_seed(19)
    k, n = 512, 768
    qweight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    wscale = torch.rand(n, 1, dtype=torch.float16) * 0.02
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    layer = LinearW4A4(k, n, output_dtype=torch.bfloat16).cuda()
    layer.load_packed_weight(pack_signed_i4(qweight), wscale, "cuda")

    actual = layer(x)
    xscale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7
    qx = (x.float() / xscale).round().clamp(-8, 7).to(torch.int32)
    # PyTorch does not provide CUDA int32 matmul; keep the exact integer
    # reference on CPU and move only the result back for scale application.
    expected = (qx.cpu() @ qweight.to(torch.int32).T).float().cuda()
    expected *= xscale * wscale.cuda().T.float()

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-1)
    assert layer.weight.numel() == n * k // 2


@CUDA
def test_public_ops_and_meta_shapes():
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    packed_x, x_scale = quantize_pack_i4(x, torch.ones(1, device="cuda", dtype=torch.float16))
    assert packed_x.shape == (3, 32)
    assert packed_x.dtype == torch.uint8
    assert x_scale.shape == (3, 1)
    assert x_scale.dtype == torch.float32

    meta_x = torch.empty(3, 64, device="meta", dtype=torch.bfloat16)
    meta_clip = torch.empty(1, device="meta", dtype=torch.float16)
    meta_packed, meta_scale = torch.ops.flatquant.quantize_pack_i4(meta_x, meta_clip)
    meta_weight = torch.empty(8, 32, device="meta", dtype=torch.uint8)
    meta_wscale = torch.empty(8, 1, device="meta", dtype=torch.float16)
    meta_out = torch.ops.flatquant.w4a4_linear(
        meta_packed, meta_weight, meta_scale, meta_wscale, torch.bfloat16
    )
    assert meta_packed.shape == (3, 32)
    assert meta_scale.shape == (3, 1)
    assert meta_out.shape == (3, 8)
    assert meta_out.dtype == torch.bfloat16


@pytest.mark.parametrize("k,n", [(65, 8), (64, 9)])
def test_constructor_rejects_unaligned_dimensions(k, n):
    with pytest.raises(ValueError, match="K % 64 == 0 and N % 8 == 0"):
        LinearW4A4(k, n)


def test_load_rejects_wrong_scale_shape():
    layer = LinearW4A4(64, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        layer.load_packed_weight(
            torch.empty(8, 32, dtype=torch.uint8), torch.empty(8), "cuda"
        )


@CUDA
def test_forward_rejects_non_bf16_activation():
    layer = LinearW4A4(64, 8).cuda()
    layer.load_packed_weight(
        torch.zeros(8, 32, dtype=torch.uint8), torch.ones(8, 1), "cuda"
    )
    with pytest.raises(TypeError, match="BF16"):
        layer(torch.zeros(1, 64, device="cuda", dtype=torch.float16))


@CUDA
def test_forward_preserves_tp_unrelated_batch_shape():
    layer = LinearW4A4(64, 8).cuda()
    layer.load_packed_weight(
        torch.zeros(8, 32, dtype=torch.uint8), torch.ones(8, 1), "cuda"
    )
    assert layer(torch.zeros(2, 3, 64, device="cuda", dtype=torch.bfloat16)).shape == (2, 3, 8)


@CUDA
def test_rejects_non_sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    with pytest.raises(NotImplementedError, match="SM80"):
        quantize_pack_i4(
            torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16),
            torch.ones(1, device="cuda", dtype=torch.float16),
        )


def test_w4a4_linear_rejects_wrong_scale_shapes_before_dispatch():
    with pytest.raises(ValueError, match="x_scale"):
        w4a4_linear(
            torch.empty(2, 32, dtype=torch.uint8),
            torch.empty(8, 32, dtype=torch.uint8),
            torch.empty(2),
            torch.empty(8, 1),
        )
