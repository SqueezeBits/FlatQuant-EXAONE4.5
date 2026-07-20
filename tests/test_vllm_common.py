import json

from benchmarks.exaone45 import vllm_common


def test_config_json_quant_method_downloads_remote_model_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"quantization_config": {"quant_method": "flatquant"}})
    )
    calls = []

    def fake_hf_hub_download(*, repo_id, filename):
        calls.append((repo_id, filename))
        return str(config_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    method = vllm_common._config_json_quant_method(
        "Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16"
    )

    assert method == "flatquant"
    assert calls == [("Hyun9junn/EXAONE-4.5-33B-FlatQuant-W4A16", "config.json")]
