import os
import sys
import unittest

import torch

from flatquant_vllm import flatquant_kron_transform

# The vLLM plugin package is a sibling directory; make it importable without
# requiring the plugin (and vLLM) to be installed in the running interpreter.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "vllm_plugin"),
)


class TritonTransformTest(unittest.TestCase):
    def test_cpu_falls_back_to_reference(self):
        x = torch.randn(2, 12)
        left = torch.randn(3, 3)
        right = torch.randn(4, 4)
        expected = x @ torch.kron(left, right)
        actual = flatquant_kron_transform(x, left, right)
        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_triton_matches_torch(self):
        # Keep the pointwise tolerance check reproducible; random near-zero
        # reference elements otherwise make this test intermittently fail.
        torch.manual_seed(0)
        for batch, left_size, right_size in ((1, 16, 16), (7, 16, 32), (2, 64, 80)):
            x = torch.randn(
                batch,
                left_size * right_size,
                device="cuda",
                dtype=torch.float16,
            )
            left = torch.randn(left_size, left_size, device="cuda", dtype=torch.float16)
            right = torch.randn(right_size, right_size, device="cuda", dtype=torch.float16)
            expected = flatquant_kron_transform(x, left, right, backend="torch")
            actual = flatquant_kron_transform(x, left, right, backend="triton")
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-1)


class FusedKronTransformTest(unittest.TestCase):
    """The fused single-launch kernel must match the two-stage reference."""

    # (left_size, right_size) for EXAONE-4.5-33B projections that carry a
    # learned right factor.
    SHAPES = ((64, 80), (128, 214))

    @staticmethod
    def _reference(x, left, right):
        shape = x.shape
        value = x.reshape(-1, left.shape[0], right.shape[0]).float()
        value = torch.matmul(value, right.float())
        value = torch.matmul(left.T.float(), value)
        return value.reshape(shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_matches_two_stage_and_reference(self):
        from flatquant_vllm_plugin.triton_transform_v2 import (
            flatquant_kron_transform as fused,
            flatquant_kron_transform_two_stage as two_stage,
        )

        torch.manual_seed(0)
        for left_size, right_size in self.SHAPES:
            for tokens in (1, 37, 512):
                x = torch.randn(
                    tokens, left_size * right_size, device="cuda", dtype=torch.bfloat16
                )
                left = torch.randn(
                    left_size, left_size, device="cuda", dtype=torch.bfloat16
                )
                right = torch.randn(
                    right_size, right_size, device="cuda", dtype=torch.bfloat16
                )
                reference = self._reference(x, left, right)
                fused_out = fused(x, left, right)
                two_stage_out = two_stage(x, left, right)
                self.assertEqual(fused_out.shape, x.shape)

                scale = reference.abs().max().clamp_min(1e-6)
                fused_rel = (fused_out.float() - reference).abs().max() / scale
                two_rel = (two_stage_out.float() - reference).abs().max() / scale
                # Fused keeps the x @ right intermediate in fp32 registers, so
                # it is never worse than the two-stage path's bf16 round-trip.
                self.assertLessEqual(fused_rel.item(), 5e-3)
                self.assertLessEqual(fused_rel.item(), two_rel.item() + 1e-4)


class KronConfigSelectionTest(unittest.TestCase):
    def test_benchmark_candidate_set_is_bounded(self):
        from benchmarks.transform_kernel_benchmark import candidate_configs
        from flatquant_vllm_plugin.triton_transform_v2 import KronConfig

        self.assertEqual(
            candidate_configs(),
            [
                KronConfig(block_n=16, num_warps=4),
                KronConfig(block_n=16, num_warps=8),
                KronConfig(block_n=32, num_warps=4),
                KronConfig(block_n=32, num_warps=8),
                KronConfig(block_n=64, num_warps=4),
                KronConfig(block_n=64, num_warps=8),
            ],
        )

    def test_selection_is_stable_for_token_bucket_boundaries(self):
        from flatquant_vllm_plugin.triton_transform_v2 import (
            KronConfig,
            select_kron_config,
        )

        allowed = {
            KronConfig(block_n=16, num_warps=4),
            KronConfig(block_n=16, num_warps=8),
            KronConfig(block_n=32, num_warps=4),
            KronConfig(block_n=32, num_warps=8),
            KronConfig(block_n=64, num_warps=4),
            KronConfig(block_n=64, num_warps=8),
        }
        for shape in ((64, 80), (128, 214), (16, 16)):
            for tokens in (1, 16, 17, 256, 257, 2048):
                first = select_kron_config(*shape, tokens, compute_capability=(8, 0))
                second = select_kron_config(*shape, tokens, compute_capability=(8, 0))
                self.assertEqual(first, second)
                self.assertIn(first, allowed)

    def test_unknown_shape_uses_documented_fallback(self):
        from flatquant_vllm_plugin.triton_transform_v2 import (
            KronConfig,
            select_kron_config,
        )

        self.assertEqual(
            select_kron_config(7, 11, 37, compute_capability=(8, 0)),
            KronConfig(block_n=64, num_warps=4),
        )

    def test_unknown_gpu_uses_documented_fallback(self):
        from flatquant_vllm_plugin.triton_transform_v2 import (
            KronConfig,
            select_kron_config,
        )

        self.assertEqual(
            select_kron_config(64, 80, 1, compute_capability=(9, 0)),
            KronConfig(block_n=64, num_warps=4),
        )

    def test_exaone_shapes_use_measured_a100_configs(self):
        from flatquant_vllm_plugin.triton_transform_v2 import (
            KronConfig,
            select_kron_config,
        )

        expected = {
            (64, 80, 1): KronConfig(16, 4),
            (64, 80, 16): KronConfig(16, 4),
            (64, 80, 64): KronConfig(32, 4),
            (64, 80, 256): KronConfig(32, 4),
            (64, 80, 512): KronConfig(32, 4),
            (64, 80, 2048): KronConfig(64, 4),
            (128, 214, 1): KronConfig(16, 4),
            (128, 214, 16): KronConfig(32, 4),
            (128, 214, 64): KronConfig(32, 4),
            (128, 214, 256): KronConfig(32, 4),
            (128, 214, 512): KronConfig(64, 4),
            (128, 214, 2048): KronConfig(64, 4),
        }
        for (left_size, right_size, tokens), config in expected.items():
            self.assertEqual(
                select_kron_config(
                    left_size,
                    right_size,
                    tokens,
                    compute_capability=(8, 0),
                ),
                config,
            )


if __name__ == "__main__":
    unittest.main()
