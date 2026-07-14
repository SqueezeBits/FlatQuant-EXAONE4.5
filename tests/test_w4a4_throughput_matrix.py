import json
from pathlib import Path

import pytest

from benchmarks.exaone45.w4a4_throughput_matrix import (
    MatrixConfig,
    render_markdown,
    validate_model_paths,
)


def test_missing_real_w4a4_artifact_fails_before_any_benchmark(tmp_path):
    bf16 = tmp_path / "bf16"
    w4a16 = tmp_path / "w4a16"
    bf16.mkdir()
    w4a16.mkdir()
    with pytest.raises(FileNotFoundError, match="real W4A4.*matrix was not run"):
        validate_model_paths(bf16, w4a16, tmp_path / "missing-w4a4")


def test_matrix_config_expands_every_controlled_case():
    config = MatrixConfig(input_lengths=(2048, 8192), concurrencies=(1, 4), output_length=32)
    assert list(config.cases()) == [(2048, 1), (2048, 4), (8192, 1), (8192, 4)]


def test_markdown_contains_metrics_counters_and_blocked_scope():
    payload = {
        "status": "verified",
        "scope": "real_33b_matrix",
        "rows": [{
            "backend": "w4a4", "input_length": 2048, "concurrency": 1,
            "prompt_tokens_per_s": 10.0, "requests_per_s": 0.1,
            "ttft_median_s": 1.0, "ttft_p95_s": 1.2,
            "peak_gpu_memory_bytes": 123, "completed_requests": 1,
            "errors": [], "dispatch_counters": {"w4a4": 4, "w4a16_fallback": 0, "bf16_fallback": 0},
        }],
    }
    text = render_markdown(payload)
    assert "prompt tokens/s" in text
    assert "TTFT p95" in text
    assert "w4a16_fallback" in text
    assert "real_33b_matrix" in text
    json.dumps(payload)
