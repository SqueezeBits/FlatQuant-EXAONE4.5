import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks.exaone45.vllm_w4a4 import compare_logits, require_local_path


def test_compare_logits_records_deterministic_numeric_deltas_and_counters():
    result = compare_logits(
        torch.tensor([[1.0, -1.0], [3.0, 5.0]]),
        torch.tensor([[0.0, -2.0], [3.0, 7.0]]),
        [4, 8, 9],
        [4, 7, 9],
        {"w4a4": 12, "fallback": 0},
    )
    assert result.mean_abs_error == pytest.approx(1.0)
    assert result.max_abs_error == pytest.approx(2.0)
    assert result.token_agreement == pytest.approx(2 / 3)
    assert result.fallback_counts == {"w4a4": 12, "fallback": 0}


def test_compare_logits_rejects_mismatched_shapes_and_empty_tokens():
    with pytest.raises(ValueError, match="shape"):
        compare_logits(torch.zeros(2), torch.zeros(3), [1], [1], {})
    with pytest.raises(ValueError, match="non-empty"):
        compare_logits(torch.zeros(2), torch.zeros(2), [], [], {})
    with pytest.raises(ValueError, match="same length"):
        compare_logits(torch.zeros(2), torch.zeros(2), [1], [1, 2], {})


def test_require_local_path_fails_clearly_instead_of_downloading(tmp_path):
    missing = tmp_path / "missing-checkpoint"
    with pytest.raises(FileNotFoundError, match="local checkpoint.*does not exist"):
        require_local_path(missing, "model")


def test_cli_missing_real_checkpoint_exits_without_claiming_success(tmp_path):
    script = Path(__file__).parents[1] / "benchmarks/exaone45/vllm_w4a4.py"
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "generate",
            "--model",
            str(tmp_path / "missing"),
            "--prompt",
            "hello",
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "local checkpoint" in result.stderr
    assert not report.exists()


def test_report_schema_is_json_serializable():
    result = compare_logits(torch.zeros(1), torch.ones(1), [1], [1], {"fallback": 0})
    payload = result.to_report(layer_tolerance=0.1, logit_tolerance=2.0)
    assert json.loads(json.dumps(payload))["fallback_counts"] == {"fallback": 0}
