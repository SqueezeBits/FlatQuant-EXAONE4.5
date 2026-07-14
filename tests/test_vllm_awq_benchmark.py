import math
from types import SimpleNamespace

from benchmarks.exaone45.vllm_awq import _engine_kwargs, summarize_latency


def test_summarize_latency_reports_ttft_tpot_and_throughput():
    samples = [
        {
            "first_token_s": 0.2,
            "elapsed_s": 1.0,
            "input_tokens": 8,
            "output_tokens": 4,
        },
        {
            "first_token_s": 0.4,
            "elapsed_s": 1.4,
            "input_tokens": 8,
            "output_tokens": 4,
        },
    ]

    result = summarize_latency(samples)

    assert result["ttft_source"] == "vllm_request_metrics"
    assert math.isclose(result["ttft_s_median"], 0.3)
    assert math.isclose(result["tpot_ms_median"], 300.0)
    assert result["e2e_s_median"] == 1.2
    assert math.isclose(result["input_tokens_per_s_median"], (8.0 + 8.0 / 1.4) / 2)
    assert math.isclose(result["output_tokens_per_s_median"], (4.0 + 4.0 / 1.4) / 2)


def test_summarize_latency_handles_missing_ttft_and_single_output_token():
    samples = [
        {
            "first_token_s": None,
            "elapsed_s": 0.5,
            "input_tokens": 8,
            "output_tokens": 1,
        },
        {
            "first_token_s": None,
            "elapsed_s": 0.25,
            "input_tokens": 8,
            "output_tokens": 0,
        },
    ]

    result = summarize_latency(samples)

    assert result["ttft_source"] == "unavailable"
    assert result["ttft_s_median"] is None
    assert result["tpot_ms_median"] is None
    assert result["output_tokens_per_s_median"] == 1.0


def test_summarize_latency_uses_per_request_tokens_for_batched_tpot():
    samples = [
        {
            "first_token_s": 0.7,
            "elapsed_s": 1.0,
            "input_tokens": 2048,
            "output_tokens": 64,
            "output_tokens_per_request": 16,
        }
    ]

    result = summarize_latency(samples)

    assert math.isclose(result["tpot_ms_median"], 20.0)
    assert result["output_tokens_per_s_median"] == 64.0


def test_summarize_latency_rejects_empty_samples():
    try:
        summarize_latency([])
    except ValueError as error:
        assert str(error) == "At least one latency sample is required."
    else:
        raise AssertionError("empty samples must be rejected")


def test_engine_kwargs_enable_request_metrics():
    args = SimpleNamespace(
        model_path="model",
        tokenizer=None,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=False,
        enable_prefix_caching=False,
    )

    kwargs = _engine_kwargs(args)

    assert kwargs["disable_log_stats"] is False
