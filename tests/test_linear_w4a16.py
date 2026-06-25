import unittest

import torch

from deploy.nn import LinearW4A16, LinearW4A16Marlin, is_marlin_available


def _pack_signed_i4(weight):
    encoded = torch.where(weight < 0, weight + 16, weight).to(torch.uint8)
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).contiguous()


class LinearW4A16Test(unittest.TestCase):
    def test_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            LinearW4A16(96, 128)
        with self.assertRaises(ValueError):
            LinearW4A16(128, 10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_matches_dequantized_reference_and_stays_packed(self):
        torch.manual_seed(7)
        rows, in_features, out_features = 9, 512, 512
        quantized = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
        packed = _pack_signed_i4(quantized)
        scale = torch.rand(out_features, 1, dtype=torch.float32) * 0.02

        linear = LinearW4A16(
            in_features,
            out_features,
            output_dtype=torch.bfloat16,
        ).cuda()
        linear.load_packed_weight(packed, scale, "cuda")

        x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
        output = linear(x)
        reference = x.float() @ (quantized.cuda().float() * scale.cuda()).T

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (rows, out_features))
        self.assertEqual(
            linear.weight.numel() * linear.weight.element_size(),
            out_features * in_features // 2,
        )
        self.assertLess((output.float() - reference).abs().mean().item(), 0.01)


class LinearW4A16MarlinTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and is_marlin_available(), "Marlin CUDA is required")
    def test_matches_reference_returns_requested_dtype_and_stays_packed(self):
        torch.manual_seed(11)
        rows, in_features, out_features = 16, 512, 512
        quantized = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
        packed = _pack_signed_i4(quantized)
        scale = torch.rand(out_features, 1, dtype=torch.float32) * 0.02

        linear = LinearW4A16Marlin(
            in_features,
            out_features,
            output_dtype=torch.bfloat16,
        ).cuda()
        linear.load_packed_weight(packed, scale, "cuda")

        x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
        output = linear(x)
        reference_weight = quantized.cuda().to(torch.float16) * scale.cuda().to(torch.float16)
        reference = torch.nn.functional.linear(x, reference_weight)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (rows, out_features))
        self.assertEqual(
            linear.weight.numel() * linear.weight.element_size(),
            out_features * in_features // 2,
        )
        relative_error = (
            (output.float() - reference.float()).abs().mean()
            / reference.float().abs().mean()
        )
        self.assertLess(relative_error.item(), 0.002)


if __name__ == "__main__":
    unittest.main()
