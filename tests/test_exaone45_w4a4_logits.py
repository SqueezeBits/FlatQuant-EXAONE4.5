import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from benchmarks.exaone45.vllm_w4a4 import (
    aggregate_worker_counters,
    compare_logits,
    require_local_path,
    query_worker_counters,
    reset_worker_counters,
    run_cuda_graph_probe,
)
from tools.export_flatquant_w4a4_vllm import export_checkpoint


def _make_tiny_conditional_checkpoint(path: Path) -> Path:
    """Build an actual HF conditional checkpoint, then replace its text linears."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        Exaone4Config,
        Exaone4_5_Config,
        Exaone4_5_ForConditionalGeneration,
        Exaone4_5_VisionConfig,
        PreTrainedTokenizerFast,
    )
    from flatquant_w4a4.packing import pack_signed_i4

    source = path / "source"
    source.mkdir(parents=True)
    text = Exaone4Config(
        vocab_size=128, hidden_size=320, intermediate_size=320,
        num_hidden_layers=1, num_attention_heads=40, num_key_value_heads=1,
        max_position_embeddings=128, sliding_window=128, sliding_window_pattern=1,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    vision = Exaone4_5_VisionConfig(
        depth=0, hidden_size=64, intermediate_size=64, num_heads=1,
        num_key_value_heads=1, out_hidden_size=320, fullatt_block_indexes=[],
    )
    config = Exaone4_5_Config(
        architectures=["Exaone4_5_ForConditionalGeneration"],
        text_config=text, vision_config=vision, image_token_id=67, video_token_id=68,
    )
    model = Exaone4_5_ForConditionalGeneration(config)
    tensors = {}
    for name, value in model.state_dict().items():
        if name.startswith("model.language_model."):
            name = "language_model.model." + name.removeprefix("model.language_model.")
        else:
            name = name.removeprefix("model.")
        if ".layers.0." not in name or not name.endswith("_proj.weight"):
            tensors[name] = value.contiguous()
    prefix = "language_model.model.layers.0."
    rows = {
        "self_attn.q_proj": 320, "self_attn.k_proj": 8,
        "self_attn.v_proj": 8, "self_attn.o_proj": 320,
        "mlp.gate_proj": 320, "mlp.up_proj": 320, "mlp.down_proj": 320,
    }
    inputs = {name: (320 if name != "mlp.down_proj" else 320) for name in rows}
    for name, out_features in rows.items():
        q = torch.zeros((out_features, inputs[name]), dtype=torch.int8)
        tensors[prefix + name + ".weight"] = pack_signed_i4(q)
        tensors[prefix + name + ".weight_scale"] = torch.ones((out_features, 1), dtype=torch.float16)
    eye = torch.eye(8, dtype=torch.bfloat16)
    tensors[prefix + "self_attn.qkv_trans.matrix_left"] = torch.eye(16, dtype=torch.bfloat16)
    tensors[prefix + "self_attn.qkv_trans.matrix_right"] = torch.eye(20, dtype=torch.bfloat16)
    tensors[prefix + "self_attn.o_trans.matrix"] = torch.eye(40, dtype=torch.bfloat16)
    tensors[prefix + "mlp.up_gate_trans.matrix_left"] = torch.eye(16, dtype=torch.bfloat16)
    tensors[prefix + "mlp.up_gate_trans.matrix_right"] = torch.eye(20, dtype=torch.bfloat16)
    tensors[prefix + "mlp.down_trans.matrix_left"] = torch.eye(16, dtype=torch.bfloat16)
    tensors[prefix + "mlp.down_trans.matrix_right"] = torch.eye(20, dtype=torch.bfloat16)
    for name in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
        tensors[prefix + name + ".activation_clip"] = torch.tensor(1.0, dtype=torch.float16)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, source / shard)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, shard)}))
    config.save_pretrained(source)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4, "<unk>": 5}
    vocab.update({f"unused_{i}": i for i in range(6, 67)})
    vocab["<|image_pad|>"] = 67
    vocab["<|video_pad|>"] = 68
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>"],
    ).save_pretrained(source)
    image_processor = {
        "image_processor_type": "Qwen2VLImageProcessor", "patch_size": 14,
        "temporal_patch_size": 2, "merge_size": 2, "min_pixels": 3136,
        "max_pixels": 3136, "do_resize": True, "do_rescale": True,
        "do_normalize": True, "do_convert_rgb": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }
    video_processor = {
        **image_processor, "video_processor_type": "Qwen2VLVideoProcessor",
        "min_frames": 4, "max_frames": 8,
    }
    (source / "preprocessor_config.json").write_text(json.dumps(image_processor))
    (source / "processor_config.json").write_text(json.dumps({
        "processor_class": "Exaone4_5_Processor",
        "image_processor": image_processor,
        "video_processor": video_processor,
    }))
    output = path / "model"
    export_checkpoint(source, output)
    return output


def test_tiny_conditional_checkpoint_loads_generates_and_dispatches_w4a4(tmp_path, monkeypatch):
    if not Path("/dev/nvidia0").exists():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    pytest.importorskip("vllm")
    model = _make_tiny_conditional_checkpoint(tmp_path)
    from benchmarks.exaone45.vllm_w4a4 import _load_llm
    from vllm import SamplingParams

    llm = _load_llm(model, True)
    reset_worker_counters(llm)
    output = llm.generate(
        [{"prompt_token_ids": [1, 3]}],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    assert len(output[0].outputs[0].token_ids) == 1
    assert query_worker_counters(llm)["w4a4"] == 4


def test_worker_counter_aggregation_rejects_an_unselected_w4a4_backend():
    assert aggregate_worker_counters([{"w4a4": 2}, {"w4a4": 3}]) == {"w4a4": 5}
    with pytest.raises(RuntimeError, match="did not dispatch"):
        aggregate_worker_counters([{}])


def test_compare_logits_records_deterministic_numeric_deltas_and_counters():
    result = compare_logits(
        torch.tensor([[1.0, -1.0], [3.0, 5.0]]),
        torch.tensor([[0.0, -2.0], [3.0, 7.0]]),
        [4, 8, 9],
        [4, 7, 9],
        {"w4a4": 12, "w4a16_fallback": 0, "bf16_fallback": 0},
    )
    assert result.mean_abs_error == pytest.approx(1.0)
    assert result.max_abs_error == pytest.approx(2.0)
    assert result.token_agreement == pytest.approx(2 / 3)
    assert result.fallback_counts == {
        "w4a4": 12, "w4a16_fallback": 0, "bf16_fallback": 0
    }


@pytest.mark.parametrize("name", ["w4a16_fallback", "bf16_fallback"])
def test_worker_counter_aggregation_rejects_each_fallback(name):
    from benchmarks.exaone45.vllm_w4a4 import aggregate_worker_counters

    with pytest.raises(RuntimeError, match=name):
        aggregate_worker_counters([{"w4a4": 1, name: 1}])


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


def test_tiny_conditional_checkpoint_cuda_graph_replays_changed_values_with_stable_memory(
    tmp_path, monkeypatch
):
    if not Path("/dev/nvidia0").exists():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("FLATQUANT_W4A4_STRICT", "1")
    pytest.importorskip("vllm")
    model = _make_tiny_conditional_checkpoint(tmp_path)

    report = run_cuda_graph_probe(
        model,
        replay_cases=[
            {"first": [1, 3], "second": [1, 4], "output_length": 1},
            {"first": [1, 3, 4], "second": [1, 4, 3], "output_length": 2},
        ],
    )

    assert report["verified_fixture"] == "tiny_conditional_w4a4"
    assert report["enforce_eager"] is False
    assert report["same_shape_changed_values"] is True
    assert report["allocated_memory_stable"] is True
    assert len(report["replays"]) == 2
    assert all(item["outputs_respond"] for item in report["replays"])
    assert report["native_generation_valid"] is True
    assert all(item["native_generation_valid"] for item in report["replays"])
    assert [item["generated_token_counts"] for item in report["replays"]] == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ]
    assert all(item["allocated_memory_stable"] for item in report["replays"])
    assert report["selection_evidence"]["w4a4_projection_count"] == 4
    assert len(report["selection_evidence"]["w4a4_projection_prefixes"]) == 4
    assert set(report["meta_kernels"]) == {
        "flatquant::quantize_pack_i4",
        "flatquant::w4a4_linear",
        "flatquant::kron_transform",
        "flatquant::left_transform",
    }
