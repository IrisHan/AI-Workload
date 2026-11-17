"""Hub scope classification and Qwen3 MoE (Coder) using existing GQA + MoE methods."""
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from app.engines.model_memory import memory_components
from app.engines.model_structure import inspect_model
from app.integrations.huggingface import HuggingFaceImporter, hub_scope
from app.lib.spec_studio.validation import shared_expert_multiplicity
from app.main import create_app
from fastapi.testclient import TestClient

QWEN = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-14-qwen-family/evidence/sources"


def source(name):
    folder = QWEN / name
    cfg = json.loads((folder / "config.json").read_text())
    meta = json.loads((folder / "metadata.json").read_text())
    return folder, cfg, meta


@contextmanager
def import_local(name, safetensors=None):
    folder, cfg, meta = source(name)
    sha = meta["sha"]
    tensors = safetensors if safetensors is not None else meta.get("safetensors") or {}
    raw = (folder / "config.json").read_bytes()

    def handler(request):
        if "/api/" in request.url.path:
            return httpx.Response(200, json={"sha": sha, "safetensors": tensors})
        return httpx.Response(200, content=raw)

    with TestClient(create_app("sqlite:///:memory:")) as client:
        client.app.state.importer = HuggingFaceImporter(httpx.MockTransport(handler))
        response = client.post("/api/models/import", json={"repo_id": meta["id"]})
        yield response, cfg, meta


def test_hub_scope_tied_duplicate_and_qwen1_bias():
    _, qwen06, meta06 = source("Qwen--Qwen3-0.6B")
    logical = inspect_model(qwen06)["totals"]["total"]
    hub = meta06["safetensors"]["total"]
    assert logical == 596049920
    kind, delta, _ = hub_scope(qwen06, logical, hub)
    assert kind == "tied-lm-head-duplicate"
    assert delta == qwen06["vocab_size"] * qwen06["hidden_size"]

    _, qwen1, meta1 = source("Qwen--Qwen-7B")
    logical1 = inspect_model(qwen1)["totals"]["total"]
    kind1, delta1, _ = hub_scope(qwen1, logical1, meta1["safetensors"]["total"])
    assert kind1 == "qwen1-fused-qkv-bias"
    assert delta1 == qwen1["num_hidden_layers"] * 3 * qwen1["hidden_size"]


def test_hub_scope_does_not_blanket_exempt_tied_models():
    _, qwen4, meta4 = source("Qwen--Qwen3-4B")
    logical = inspect_model(qwen4)["totals"]["total"]
    kind, delta, _ = hub_scope(qwen4, logical, meta4["safetensors"]["total"])
    assert qwen4["tie_word_embeddings"] is True
    assert kind == "same-scope"
    assert delta == 0
    with pytest.raises(ValueError, match="Parameter count mismatch"):
        hub_scope(qwen4, logical, logical + 7)


def test_import_qwen1_and_tied_small_qwen3():
    for name, comparison in (
        ("Qwen--Qwen-7B", "qwen1-fused-qkv-bias"),
        ("Qwen--Qwen-14B", "qwen1-fused-qkv-bias"),
        ("Qwen--Qwen3-0.6B", "tied-lm-head-duplicate"),
        ("Qwen--Qwen3-1.7B", "tied-lm-head-duplicate"),
    ):
        with import_local(name) as (response, cfg, meta):
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["source_metadata"]["comparison"] == comparison
            assert body["parameters"] == inspect_model(cfg)["totals"]["total"]
            assert body["source_metadata"]["checkpoint_minus_main"] == (
                meta["safetensors"]["total"] - body["parameters"]
            )


def test_shared_width_three_state():
    assert shared_expert_multiplicity(0, 2560, 0) == 0
    assert shared_expert_multiplicity(2560, 2560, 0) == 1
    assert shared_expert_multiplicity(5120, 2560, 0) == 2
    assert shared_expert_multiplicity(None, 2560, 1) == 1
    with pytest.raises(ValueError, match="independent shared expert width"):
        shared_expert_multiplicity(3000, 2560, 0)
    with pytest.raises(ValueError, match="conflicts with zero"):
        shared_expert_multiplicity(0, 2560, 1)


def test_coder_gqa_moe_matches_hub_and_has_no_shared_expert():
    _, cfg, meta = source("Qwen--Qwen3-Coder-480B-A35B-Instruct")
    result = inspect_model(cfg)
    assert result["totals"]["total"] == meta["safetensors"]["total"] == 480154875392
    assert result["templates"][0]["attn_kind"] == "GQA"
    assert result["templates"][0]["ffn_kind"] == "MOE"
    assert sum(g["count"] for g in result["groups"]) == 62
    moe_layers = sum(
        g["count"] for g in result["groups"]
        if next(t for t in result["templates"] if t["id"] == g["template_id"])["ffn_kind"] == "MOE"
    )
    assert moe_layers == 62
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "shared_expert.gate_proj" not in names
    assert "routed_expert.gate_proj" in names
    assert "q_norm" in names
    with import_local("Qwen--Qwen3-Coder-480B-A35B-Instruct") as (response, _, _):
        assert response.status_code == 200, response.text
        assert response.json()["source_metadata"]["comparison"] == "same-scope"
        assert response.json()["is_moe"] is True


def _group_counts(result):
    counts = {}
    for group in result["groups"]:
        template = next(t for t in result["templates"] if t["id"] == group["template_id"])
        counts[(template["attn_kind"], template["ffn_kind"])] = (
            counts.get((template["attn_kind"], template["ffn_kind"]), 0) + group["count"]
        )
    return counts


def test_next_hybrid_moe_interval_and_hub_same_scope():
    _, cfg, meta = source("Qwen--Qwen3-Next-80B-A3B-Instruct")
    result = inspect_model(cfg)
    assert _group_counts(result) == {("GATED_DELTA", "MOE"): 36, ("GQA_GATE", "MOE"): 12}
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_gate_proj" in names
    assert "in_proj_qkv" in names
    assert "shared_expert.gate_proj" in names
    assert "routed_expert.gate_proj" in names
    assert result["identity"]["model_type"] == "qwen3_next"
    assert result["totals"].get("vision") in (None, 0)
    assert "shared_expert_gate" in names
    assert result["totals"]["total"] != meta["safetensors"]["total"]
    mem = memory_components(result, cfg, context_length=128)
    mem256 = memory_components(result, cfg, context_length=256)
    assert mem256["kv_cache_bytes"] - mem["kv_cache_bytes"] == 12 * 128 * 2 * 2 * 256 * 2
    assert mem["cache_components"][0]["recurrent_state_bytes"] == 32 * 128 * 128 * 4
    with import_local("Qwen--Qwen3-Next-80B-A3B-Instruct") as (response, _, _):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source_metadata"]["comparison"] == "different-scope"
        assert body["is_moe"] is True
        assert body["parameter_scope"] == "main"
        assert body["parameters"] == result["totals"]["total"]
        assert body["source_metadata"]["checkpoint_minus_main"] == (
            meta["safetensors"]["total"] - body["parameters"]
        )


def test_qwen36_a3b_hybrid_moe_with_optional_vision():
    _, cfg, meta = source("Qwen--Qwen3.6-35B-A3B")
    result = inspect_model(cfg)
    assert _group_counts(result) == {("GATED_DELTA", "MOE"): 30, ("GQA_GATE", "MOE"): 10}
    assert result["totals"]["vision"]
    assert "intermediate_size" not in cfg["text_config"]
    assert result["identity"]["model_type"] == "qwen3_5_moe"
    with_mtp = inspect_model(cfg, include_mtp=True)
    assert with_mtp["totals"]["total"] == meta["safetensors"]["total"]
    with import_local("Qwen--Qwen3.6-35B-A3B") as (response, _, _):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["parameter_scope"] == "main-with-vision"
        assert body["source_metadata"]["comparison"] == "different-scope"
        assert body["parameters"] == result["totals"]["total"]
        assert body["source_metadata"]["checkpoint_minus_main"] == (
            meta["safetensors"]["total"] - body["parameters"]
        )


def test_qwen38_24t_text_only_hybrid_moe():
    _, cfg, meta = source("Qwen--Qwen3.8-2.4T-A95B")
    result = inspect_model(cfg)
    assert _group_counts(result) == {("GATED_DELTA", "MOE"): 69, ("GQA_GATE", "MOE"): 23}
    assert not result["totals"].get("vision")
    assert cfg.get("vision_config") is None
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_gate_proj" in names
    with_mtp = inspect_model(cfg, include_mtp=True)
    assert with_mtp["totals"]["total"] == meta["safetensors"]["total"]
    with import_local("Qwen--Qwen3.8-2.4T-A95B") as (response, _, _):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["parameter_scope"] == "main"
        assert body["source_metadata"]["comparison"] == "different-scope"
        assert body["is_moe"] is True
        assert body["parameters"] == result["totals"]["total"]
        assert body["source_metadata"]["checkpoint_minus_main"] == (
            meta["safetensors"]["total"] - body["parameters"]
        )


def test_flash_next_rejected_for_qsa_ngram_and_hc_lowrank():
    for name in ("Qwen--Qwen3.8-Flash-Next", "Qwen--Qwen3.8-Flash-Next-FP8"):
        cfg = json.loads((QWEN / name / "config.json").read_text())
        with pytest.raises(ValueError, match="QSA indexer"):
            inspect_model(cfg)
        with pytest.raises(ValueError, match="n-gram/PLE"):
            inspect_model(cfg)
        with pytest.raises(ValueError, match="hc_count/hc_lowrank"):
            inspect_model(cfg)
        with import_local(name) as (response, _, _):
            assert response.status_code == 422
            assert "QSA indexer" in response.text
            assert "n-gram" in response.text
            assert "hc_lowrank" in response.text


def test_quant_payload_is_not_bf16_times_two():
    _, fp8, _ = source("Qwen--Qwen3-32B-FP8")
    result = inspect_model(fp8)
    bf16 = inspect_model(json.loads((QWEN / "Qwen--Qwen3-32B" / "config.json").read_text()))
    assert result["totals"]["total"] == bf16["totals"]["total"] == 32762123264
    mem = memory_components(result, fp8)
    bf16_mem = memory_components(bf16, json.loads((QWEN / "Qwen--Qwen3-32B" / "config.json").read_text()))
    assert mem["weight_bytes"] >= result["totals"]["total"]
    assert mem["weight_bytes"] < bf16_mem["weight_bytes"]
    assert mem["formula_version"] == "static-memory-components-1.3.0"
    assert bf16_mem["weight_bytes"] == result["totals"]["total"] * 2
    _, awq, _ = source("Qwen--Qwen3-32B-AWQ")
    awq_result = inspect_model(awq)
    assert awq_result["totals"]["total"] == 32762123264
    awq_mem = memory_components(awq_result, awq)
    assert awq_mem["weight_bytes"] > 32762123264 // 2
    assert awq_mem["weight_bytes"] < 32762123264
    _, hybrid_fp8, _ = source("Qwen--Qwen3.5-27B-FP8")
    hybrid = inspect_model(hybrid_fp8)
    assert hybrid["totals"]["total"] == 27356728560
    hybrid_mem = memory_components(hybrid, hybrid_fp8)
    assert hybrid_mem["weight_bytes"] >= 27356728560
    assert hybrid_mem["weight_bytes"] < 27356728560 * 2


def _node(template, node_id):
    return next(node for node in template["nodes"] if node["id"] == node_id)


def test_dense_qwen_ffn_graph_has_no_router():
    _, cfg, _ = source("Qwen--Qwen2.5-0.5B")
    result = inspect_model(cfg)
    template = result["templates"][0]
    assert template["ffn_kind"] == "DENSE"
    assert template.get("moe") is None
    ids = {node["id"] for node in template["nodes"]}
    assert "router" not in ids
    assert "routed_expert.gate_proj" not in ids
    assert "gate_proj" in ids
    assert _node(template, "gate_proj")["label"] == "gate_proj"
    assert _node(template, "up_proj")["label"] == "up_proj"
    assert "cluster" not in _node(template, "gate_proj")


def test_coder_moe_graph_shows_routed_multiplicity_without_shared():
    _, cfg, _ = source("Qwen--Qwen3-Coder-480B-A35B-Instruct")
    result = inspect_model(cfg)
    template = next(item for item in result["templates"] if item["ffn_kind"] == "MOE")
    moe = template["moe"]
    assert moe["n_routed"] == 160
    assert moe["top_k"] == 8
    assert moe["n_shared"] == 0
    assert moe["has_shared_expert_gate"] is False
    gate = _node(template, "routed_expert.gate_proj")
    assert gate["cluster"] == "routed"
    assert "×160" in gate["label"]
    assert " · gate" in gate["label"]
    assert _node(template, "router")["cluster"] == "router"
    ids = {node["id"] for node in template["nodes"]}
    assert "shared_expert.gate_proj" not in ids
    assert "gate_proj" not in ids or "routed_expert.gate_proj" in ids


def test_next_moe_graph_shows_shared_and_shared_gate():
    _, cfg, _ = source("Qwen--Qwen3-Next-80B-A3B-Instruct")
    result = inspect_model(cfg)
    template = next(item for item in result["templates"] if item["ffn_kind"] == "MOE")
    moe = template["moe"]
    assert moe["n_routed"] == 512
    assert moe["top_k"] == 10
    assert moe["n_shared"] == 1
    assert moe["shared_width"] == moe["expert_dim"] == 512
    assert moe["has_shared_expert_gate"] is True
    assert "×512" in _node(template, "routed_expert.gate_proj")["label"]
    assert "×1" in _node(template, "shared_expert.gate_proj")["label"]
    assert _node(template, "shared_expert.gate_proj")["cluster"] == "shared"
    assert "d→1" in _node(template, "shared_expert_gate")["label"]
    ids = {node["id"] for node in template["nodes"]}
    assert "routed_expert.silu" in ids
    assert "routed_expert.multiply" in ids
    assert "shared_expert.silu" in ids


def test_v4_moe_graph_keeps_totals_and_shows_expert_columns():
    cfg = json.loads(
        (Path(__file__).resolve().parents[2]
         / "docs/reports/2026-09-13-model-center/evidence/sources"
         / "deepseek-ai--DeepSeek-V4-Flash-Base/config.json").read_text()
    )
    result = inspect_model(cfg)
    assert result["totals"]["total"] == 284332240471
    template = result["templates"][0]
    assert template["ffn_kind"] == "MOE"
    moe = template["moe"]
    assert moe["n_routed"] == cfg["n_routed_experts"]
    assert moe["n_shared"] == 1
    assert "×" in _node(template, "routed_expert.gate_proj")["label"]
    assert _node(template, "shared_expert.gate_proj")["cluster"] == "shared"
    assert _node(template, "router")["cluster"] == "router"
