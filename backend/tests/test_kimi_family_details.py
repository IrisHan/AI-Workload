"""Kimi family: method authorization, MoonViT, LatentMoE/KDA — not per-repo formulas."""
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from app.engines.model_memory import memory_components
from app.engines.model_structure import inspect_model
from app.integrations.huggingface import HuggingFaceImporter, hub_scope
from app.main import create_app
from fastapi.testclient import TestClient

KIMI = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-15-kimi-family/evidence/sources"
MC = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-13-model-center/evidence/sources"


def source(name):
    folder = KIMI / name
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


def _group_counts(result):
    counts = {}
    for group in result["groups"]:
        template = next(t for t in result["templates"] if t["id"] == group["template_id"])
        counts[(template["attn_kind"], template["ffn_kind"])] = (
            counts.get((template["attn_kind"], template["ffn_kind"]), 0) + group["count"]
        )
    return counts


def test_deepseek_v3_gold_unchanged():
    cfg = json.loads((MC / "deepseek-ai--DeepSeek-V3-Base" / "config.json").read_text())
    result = inspect_model(cfg)
    assert result["totals"]["total"] == 671026419200
    assert any(t["attn_kind"] == "MLA" and t["ffn_kind"] == "MOE" for t in result["templates"])


def test_k2_instruct_mla_moe_dense_prefix():
    _, cfg, meta = source("moonshotai--Kimi-K2-Instruct")
    result = inspect_model(cfg)
    assert _group_counts(result) == {("MLA", "DENSE"): 1, ("MLA", "MOE"): 60}
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_a_proj" in names
    assert "kv_a_proj" in names
    assert "g_proj" not in names
    assert "routed_expert.gate_proj" in names
    kind, _, _ = hub_scope(cfg, result["totals"]["total"], meta["safetensors"]["total"])
    assert kind == "quant-scale"
    mem = memory_components(result, cfg)
    assert mem["weight_bytes"] >= result["totals"]["total"]
    assert mem["weight_bytes"] < 2 * result["totals"]["total"]
    with import_local("moonshotai--Kimi-K2-Instruct") as (response, _, _):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["parameters"] == result["totals"]["total"]
        assert body["is_moe"] is True
        assert body["workload_supported"] is True


def test_vl_language_plus_moonvit():
    _, cfg, meta = source("moonshotai--Kimi-VL-A3B-Instruct")
    result = inspect_model(cfg)
    assert _group_counts(result) == {("MLA", "DENSE"): 1, ("MLA", "MOE"): 26}
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_proj" in names
    assert "q_a_proj" not in names
    vision_ids = {row["id"] for row in result["shared_modules"] if row["id"] == "vision"}
    assert vision_ids
    assert result["totals"]["vision"]
    assert result["identity"]["model_type"] == "kimi_vl"
    kind, _, _ = hub_scope(cfg, result["totals"]["total"], meta["safetensors"]["total"])
    assert kind == "different-scope"
    with import_local("moonshotai--Kimi-VL-A3B-Instruct") as (response, _, _):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["parameter_scope"] == "main-with-vision"
        assert body["parameters"] == result["totals"]["total"]


def test_k25_family_isomorphic_language_plus_moonvit():
    totals = []
    for name in (
        "moonshotai--Kimi-K2.5",
        "moonshotai--Kimi-K2.6",
        "moonshotai--Kimi-K2.7-Code",
    ):
        _, cfg, meta = source(name)
        result = inspect_model(cfg)
        assert _group_counts(result) == {("MLA", "DENSE"): 1, ("MLA", "MOE"): 60}
        assert result["totals"]["vision"]
        totals.append(result["totals"]["total"])
        kind, _, _ = hub_scope(cfg, result["totals"]["total"], meta["safetensors"]["total"])
        assert kind == "quant-scale"
        mem = memory_components(result, cfg)
        assert mem["weight_bytes"] > result["totals"]["total"] // 2
        assert mem["weight_bytes"] < result["totals"]["total"]
        assert mem["formula_version"] == "static-memory-components-1.3.0"
        with import_local(name) as (response, _, _):
            assert response.status_code == 200, response.text
            assert response.json()["parameters"] == result["totals"]["total"]
    assert len(set(totals)) == 1


def test_k3_kda_gated_mla_latent_moe_attnres():
    _, cfg, meta = source("moonshotai--Kimi-K3")
    result = inspect_model(cfg)
    assert _group_counts(result) == {
        ("KDA", "DENSE"): 1,
        ("KDA", "MOE"): 68,
        ("MLA", "MOE"): 24,
    }
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_conv1d" in names
    assert "g_proj" in names
    assert "routed_expert_down_proj" in names
    assert "routed_expert_up_proj" in names
    assert "self_attention_res_proj" in names
    assert "shared_expert_gate" not in names
    shared = {row["id"] for row in result["shared_modules"]}
    assert "output_attn_res_proj" in shared
    assert result["totals"]["vision"]
    assert result["totals"]["total"] < 2 ** 53
    mem = memory_components(result, cfg, context_length=128)
    mem256 = memory_components(result, cfg, context_length=256)
    assert mem256["kv_cache_bytes"] > mem["kv_cache_bytes"]
    kind, _, _ = hub_scope(cfg, result["totals"]["total"], meta["safetensors"]["total"])
    assert kind == "quant-scale"
    with import_local("moonshotai--Kimi-K3") as (response, _, _):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["is_moe"] is True
        assert body["workload_supported"] is True
        assert body["parameters"] == result["totals"]["total"]


def test_k3_without_kda_schedule_is_rejected():
    _, cfg, _ = source("moonshotai--Kimi-K3")
    broken = json.loads(json.dumps(cfg))
    broken["text_config"].pop("linear_attn_config")
    with pytest.raises(ValueError, match="unsupported layer variants"):
        inspect_model(broken)
