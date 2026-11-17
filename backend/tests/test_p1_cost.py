"""Independent state/window goldens for MC-02/03/04/17."""
import json
from pathlib import Path

import pytest
from app.engines.model_memory import memory_components
from app.engines.model_structure import inspect_model
from app.lib.spec_studio.cost import operator_cost

ROOT = Path(__file__).resolve().parents[2]
HARDWARE = dict(peak_tflops=312, bandwidth_bytes_s=2e12, memory_bytes=80 * 1024**3,
                source="mock://gpu", version="mock-p1")


def config(group, repo):
    return json.loads((ROOT / f"docs/reports/{group}/evidence/sources/{repo}/config.json").read_text())


def cost(analysis, phase="decode", batch=1, length=128, dtype="bf16"):
    return operator_cost(analysis, phase=phase, batch_size=batch, sequence_length=length,
                         context_length=length, dtype=dtype, hardware=HARDWARE)


@pytest.mark.parametrize("group,repo", [
    ("2026-09-15-kimi-family", "moonshotai--Kimi-K3"),
    ("2026-09-14-qwen-family", "Qwen--Qwen3-Next-80B-A3B-Instruct"),
    ("2026-09-15-frontier-families", "zai-org--GLM-5.3-Flash"),
])
def test_recurrent_true_dimensions_tokens_and_residency(group, repo):
    cfg = config(group, repo)
    text = cfg.get("text_config", cfg)
    if "linear_attn_config" in text:
        spec = text["linear_attn_config"]
        heads, key, value = spec["num_heads"], spec["head_dim"], spec["head_dim"]
        conv_elements = 3 * heads * key * (spec["short_conv_kernel_size"] - 1)
    else:
        heads, key, value = text["linear_num_value_heads"], text["linear_key_head_dim"], text["linear_value_head_dim"]
        conv_elements = (2 * text["linear_num_key_heads"] * key + heads * value) * text["linear_conv_kernel_dim"]
    analysis = inspect_model(cfg)
    for phase, batch, length in [("decode", 1, 1), ("decode", 2, 128), ("prefill", 1, 128),
                                 ("prefill", 2, 4096)]:
        actual = cost(analysis, phase, batch, length)
        tokens = batch * (length if phase == "prefill" else 1)
        for row in actual["rows"]:
            if row["operation"] == "recurrent_state":
                assert row["flops"] == 2 * tokens * heads * key * value * row["count"]
        assert actual["kv_cache_bytes"] == memory_components(
            analysis, cfg, batch_size=batch, context_length=length)["kv_cache_bytes"]
        linear_count = sum(g["count"] for g in analysis["groups"] if any(
            t["id"] == g["template_id"] and t["attn_kind"] in {"KDA", "GATED_DELTA"}
            for t in analysis["templates"]))
        fixed = batch * linear_count * (heads * key * value * 4 + conv_elements * 2)
        assert actual["kv_cache_bytes"] >= fixed


@pytest.mark.parametrize("length", [64, 128, 4096, 8192])
def test_swa_decode_saturates_and_cache_agrees(length):
    cfg = config("2026-09-15-frontier-families", "openai--gpt-oss-120b")
    analysis = inspect_model(cfg)
    actual = cost(analysis, length=length)
    groups = {g["id"]: g for g in analysis["groups"]}
    for row in actual["rows"]:
        if row["operation"] != "qk_matmul":
            continue
        group = groups[row["group_id"]]
        t = next(t for t in analysis["templates"] if t["id"] == group["template_id"])
        bb = t["backbone"]
        pairs = min(length, 128) if t["attn_kind"] == "SWA" else length
        assert row["flops"] == 2 * bb["num_heads"] * bb["head_dim"] * pairs * group["count"]
    for phase in ("decode", "prefill"):
        assert cost(analysis, phase, length=length)["kv_cache_bytes"] == memory_components(
            analysis, cfg, context_length=length)["kv_cache_bytes"]


def test_old_recurrent_snapshot_requires_new_layout():
    analysis = inspect_model(config("2026-09-15-kimi-family", "moonshotai--Kimi-K3"))
    for t in analysis["templates"]:
        t.pop("cache", None)
        t.pop("recurrent", None)
    with pytest.raises(ValueError, match="create a new structure snapshot"):
        cost(analysis)
