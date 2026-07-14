import pytest
import torch

from flatquant_w4a4.packing import pack_signed_i4
from deploy.nn import LinearW4A4, quantize_pack_i4, w4a4_linear


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# LGAI-EXAONE/EXAONE-4.5-33B text_config: hidden_size=5120,
# intermediate_size=27392, num_attention_heads=40,
# num_key_value_heads=8, and therefore head_dim=128.  These are the four
# distinct dense projection shapes (Q/O, K/V, gate/up, down).
EXAONE_PROJECTION_SHAPES = (
    (5120, 5120),
    (1024, 5120),
    (27392, 5120),
    (5120, 27392),
)


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
@pytest.mark.parametrize("n,k", EXAONE_PROJECTION_SHAPES)
def test_w4a4_exaone_projection_epilogue(n, k):
    torch.manual_seed(23)
    rows = 1
    qweight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    packed_w = pack_signed_i4(qweight).cuda()
    wscale = (torch.rand(n, 1, dtype=torch.float16) * 0.02).cuda()
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    packed_x, xscale = quantize_pack_i4(
        x, torch.ones(1, device="cuda", dtype=torch.float16)
    )
    actual = torch.ops.flatquant.w4a4_linear(
        packed_x, packed_w, xscale, wscale, torch.bfloat16
    )
    qx = (x.float() / xscale).round().clamp(-8, 7).to(torch.int32)
    expected = (qx.cpu() @ qweight.to(torch.int32).T).float().cuda()
    expected *= xscale * wscale.T.float()
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-1)


@CUDA
def test_w4a4_epilogue_does_not_allocate_int32_output_matrix():
    # Peak allocated bytes are allocator-backed and include transient tensors,
    # unlike post-call snapshots.  Warm first so module/kernel initialization is
    # excluded.  A BF16 output is 2*M*N bytes; the old epilogue additionally
    # allocated a 4*M*N-byte INT32 matrix.
    m, n, k = 512, 5120, 5120
    packed_x = torch.zeros(m, k // 2, device="cuda", dtype=torch.uint8)
    packed_w = torch.zeros(n, k // 2, device="cuda", dtype=torch.uint8)
    xscale = torch.ones(m, 1, device="cuda")
    wscale = torch.ones(n, 1, device="cuda", dtype=torch.float16)
    torch.ops.flatquant.w4a4_linear(
        packed_x, packed_w, xscale, wscale, torch.bfloat16
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = torch.ops.flatquant.w4a4_linear(
        packed_x, packed_w, xscale, wscale, torch.bfloat16
    )
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    output_bytes = output.numel() * output.element_size()
    assert output.dtype == torch.bfloat16
    assert peak_delta < output_bytes + m * n * 2


@CUDA
def test_w4a4_candidate_selector_parity_and_rejection():
    m, n, k = 16, 768, 512
    px = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    pw = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    xs = torch.rand(m, 1, device="cuda")
    ws = torch.rand(n, 1, device="cuda", dtype=torch.float16)
    outputs = [torch.ops.flatquant._w4a4_linear_candidate(
        px, pw, xs, ws, torch.bfloat16, candidate) for candidate in range(3)]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="candidate"):
        torch.ops.flatquant._w4a4_linear_candidate(
            px, pw, xs, ws, torch.bfloat16, 3)
    assert [torch.ops.flatquant._w4a4_candidate_name(i) for i in range(3)] == [
        "small", "medium", "large"]


def test_w4a4_kernel_table_boundaries_and_fallback():
    tables = {
        (5120, 5120): [0, 0, 0, 0, 1],
        (1024, 5120): [0, 0, 0, 0, 0],
        (27392, 5120): [0, 0, 1, 1, 1],
        (5120, 27392): [0, 0, 0, 1, 1],
    }
    names = [
        "sm80_64x128x128_w32x64x128_s3",
        "sm80_128x128x128_w64x64x128_s3",
        "sm80_128x256x128_w64x64x128_s3",
    ]
    checks = [(31, 0), (32, 1), (127, 1), (128, 2), (511, 2),
              (512, 3), (2047, 3), (2048, 4)]
    for (n, k), table in tables.items():
        for m, bucket in checks:
            assert torch.ops.flatquant.w4a4_kernel_name(m, n, k) == names[table[bucket]]
    fallback = [0, 0, 1, 2, 2]
    for m, bucket in checks:
        assert torch.ops.flatquant.w4a4_kernel_name(m, 768, 512) == names[fallback[bucket]]


@CUDA
def test_w4a4_bias_is_fused_and_numerically_correct():
    m, n, k = 512, 768, 512
    px = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    pw = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    xs = torch.rand(m, 1, device="cuda") * 0.02
    ws = torch.rand(n, 1, device="cuda", dtype=torch.float16) * 0.02
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    base = torch.ops.flatquant.w4a4_linear(px, pw, xs, ws, torch.bfloat16)
    expected = (base.float() + bias.float()).to(torch.bfloat16)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    actual = torch.ops.flatquant.w4a4_linear(px, pw, xs, ws, torch.bfloat16, bias)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline == actual.numel() * actual.element_size()
    # Fused bias is added to FP32-scaled accumulators before the single BF16
    # rounding, whereas ``expected`` necessarily rounds the unbiased output
    # first.  Bound the resulting one-ULP difference.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-2)


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


def test_direct_quantize_rejects_cpu_before_cuda_runtime_query():
    with pytest.raises(RuntimeError, match="CUDA"):
        torch.ops.flatquant.quantize_pack_i4(
            torch.zeros(1, 64, dtype=torch.bfloat16), torch.ones(1, dtype=torch.float16)
        )


@CUDA
def test_direct_quantize_rejects_noncontiguous_input():
    x = torch.zeros(64, 2, device="cuda", dtype=torch.bfloat16).t()
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.flatquant.quantize_pack_i4(
            x, torch.ones(1, device="cuda", dtype=torch.float16)
        )


@CUDA
def test_direct_linear_rejects_noncontiguous_and_mixed_device_inputs():
    packed_x = torch.zeros(2, 64, device="cuda", dtype=torch.uint8)[:, ::2]
    packed_w = torch.zeros(8, 32, device="cuda", dtype=torch.uint8)
    xscale = torch.ones(2, 1, device="cuda")
    wscale = torch.ones(8, 1, device="cuda", dtype=torch.float16)
    assert not packed_x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.flatquant.w4a4_linear(
            packed_x, packed_w, xscale, wscale, torch.bfloat16
        )
    with pytest.raises(RuntimeError, match="CUDA|device"):
        torch.ops.flatquant.w4a4_linear(
            packed_x.contiguous(), packed_w.cpu(), xscale, wscale, torch.bfloat16
        )


@pytest.mark.parametrize(
    "x,clip,match",
    [
        (torch.empty(1, 64, device="meta", dtype=torch.float16),
         torch.empty(1, device="meta", dtype=torch.float16), "BF16"),
        (torch.empty(1, 64, device="meta", dtype=torch.bfloat16),
         torch.empty(2, device="meta", dtype=torch.float16), "clip"),
        (torch.empty(1, 66, device="meta", dtype=torch.bfloat16),
         torch.empty(1, device="meta", dtype=torch.float16), "K % 64"),
        (torch.empty(64, 2, device="meta", dtype=torch.bfloat16).t(),
         torch.empty(1, device="meta", dtype=torch.float16), "contiguous"),
    ],
)
def test_quantize_meta_rejection_parity(x, clip, match):
    with pytest.raises(RuntimeError, match=match):
        torch.ops.flatquant.quantize_pack_i4(x, clip)


@pytest.mark.parametrize(
    "packed_x,packed_w,xscale,wscale,dtype,match",
    [
        (torch.empty(2, 32, device="meta", dtype=torch.int8),
         torch.empty(8, 32, device="meta", dtype=torch.uint8),
         torch.empty(2, 1, device="meta"), torch.empty(8, 1, device="meta", dtype=torch.float16),
         torch.bfloat16, "uint8"),
        (torch.empty(2, 33, device="meta", dtype=torch.uint8),
         torch.empty(8, 33, device="meta", dtype=torch.uint8),
         torch.empty(2, 1, device="meta"), torch.empty(8, 1, device="meta", dtype=torch.float16),
         torch.bfloat16, "K % 64"),
        (torch.empty(2, 32, device="meta", dtype=torch.uint8),
         torch.empty(9, 32, device="meta", dtype=torch.uint8),
         torch.empty(2, 1, device="meta"), torch.empty(9, 1, device="meta", dtype=torch.float16),
         torch.bfloat16, "N % 8"),
        (torch.empty(2, 32, device="meta", dtype=torch.uint8),
         torch.empty(8, 32, device="meta", dtype=torch.uint8),
         torch.empty(2, 1, device="meta"), torch.empty(8, 1, device="meta", dtype=torch.float16),
         torch.float32, "output_dtype"),
        (torch.empty(2, 32, device="meta", dtype=torch.uint8),
         torch.empty(8, 32, device="meta", dtype=torch.uint8),
         torch.empty(2, 1, device="meta"), torch.empty(8, 1, device="meta", dtype=torch.float16),
         torch.float16, "BF16"),
    ],
)
def test_linear_meta_rejection_parity(packed_x, packed_w, xscale, wscale, dtype, match):
    with pytest.raises(RuntimeError, match=match):
        torch.ops.flatquant.w4a4_linear(packed_x, packed_w, xscale, wscale, dtype)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda values: values.__setitem__(0, values[0].t()), "contiguous"),
        (lambda values: values.__setitem__(1, values[1].t()), "contiguous"),
        (lambda values: values.__setitem__(2, values[2].to(torch.float16)), "FP32"),
        (lambda values: values.__setitem__(2, torch.empty(3, 1, device="meta")), "x_scale"),
        (lambda values: values.__setitem__(3, values[3].to(torch.float32)), "FP16"),
        (lambda values: values.__setitem__(3, torch.empty(7, 1, device="meta", dtype=torch.float16)), "w_scale"),
        (lambda values: values.__setitem__(5, torch.empty(8, device="meta", dtype=torch.float16)), "bias.*BF16"),
        (lambda values: values.__setitem__(5, torch.empty(16, device="meta", dtype=torch.bfloat16)[::2]), "bias.*contiguous"),
        (lambda values: values.__setitem__(5, torch.empty(7, device="meta", dtype=torch.bfloat16)), "bias.*shape"),
    ],
)
def test_linear_fake_rejects_every_eager_cuda_contract_violation(mutate, match):
    values = [
        torch.empty(32, 2, device="meta", dtype=torch.uint8).t(),
        torch.empty(32, 8, device="meta", dtype=torch.uint8).t(),
        torch.empty(2, 1, device="meta", dtype=torch.float32),
        torch.empty(8, 1, device="meta", dtype=torch.float16),
        torch.bfloat16,
        None,
    ]
    # Start contiguous; individual cases introduce one invalid property.
    values[0] = values[0].contiguous()
    values[1] = values[1].contiguous()
    mutate(values)
    with pytest.raises(RuntimeError, match=match):
        torch.ops.flatquant.w4a4_linear(*values)


@CUDA
@pytest.mark.parametrize(
    "case,match",
    [
        ("xscale_dtype", "FP32"), ("xscale_shape", "x_scale"),
        ("wscale_dtype", "FP16"), ("wscale_shape", "w_scale"),
        ("bias_dtype", "bias.*BF16"), ("bias_shape", "bias.*shape"),
        ("bias_contiguous", "bias.*contiguous"), ("input_contiguous", "contiguous"),
    ],
)
def test_fake_and_eager_cuda_reject_the_same_linear_contract(case, match):
    def values(device):
        result = [
            torch.empty(2, 32, device=device, dtype=torch.uint8),
            torch.empty(8, 32, device=device, dtype=torch.uint8),
            torch.empty(2, 1, device=device, dtype=torch.float32),
            torch.empty(8, 1, device=device, dtype=torch.float16),
            torch.bfloat16,
            None,
        ]
        if case == "xscale_dtype": result[2] = result[2].to(torch.float16)
        if case == "xscale_shape": result[2] = torch.empty(3, 1, device=device)
        if case == "wscale_dtype": result[3] = result[3].to(torch.float32)
        if case == "wscale_shape": result[3] = torch.empty(7, 1, device=device, dtype=torch.float16)
        if case == "bias_dtype": result[5] = torch.empty(8, device=device, dtype=torch.float16)
        if case == "bias_shape": result[5] = torch.empty(7, device=device, dtype=torch.bfloat16)
        if case == "bias_contiguous": result[5] = torch.empty(16, device=device, dtype=torch.bfloat16)[::2]
        if case == "input_contiguous": result[0] = torch.empty(32, 2, device=device, dtype=torch.uint8).t()
        return result

    for device in ("meta", "cuda"):
        with pytest.raises(RuntimeError, match=match):
            torch.ops.flatquant.w4a4_linear(*values(device))


@CUDA
def test_empty_rows_return_empty_without_launch():
    layer = LinearW4A4(64, 8).cuda()
    layer.load_packed_weight(
        torch.zeros(8, 32, dtype=torch.uint8), torch.ones(8, 1), "cuda"
    )
    output = layer(torch.empty(0, 64, device="cuda", dtype=torch.bfloat16))
    assert output.shape == (0, 8)
