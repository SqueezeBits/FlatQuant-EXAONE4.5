import math

from benchmarks.exaone45.vllm_awq import summarize_latency


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


def test_summarize_latency_rejects_empty_samples():
    try:
        summarize_latency([])
    except ValueError as error:
        assert str(error) == "At least one latency sample is required."
    else:
        raise AssertionError("empty samples must be rejected")
