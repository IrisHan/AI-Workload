"""Morphology tags drive cost/memory/graph — not per-repo formulas."""
import json
from pathlib import Path

import httpx
import pytest
from app.engines.model_memory import memory_components, module_payload_bytes, quantization_config
from app.engines.model_structure import inspect_model
from app.engines.morphology import module_ignored
from app.integrations.huggingface import HuggingFaceImporter
from app.lib.spec_studio.cost import operator_cost
from app.main import create_app
from fastapi.testclient import TestClient

KIMI = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-15-kimi-family/evidence/sources"
MC = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-13-model-center/evidence/sources"
QWEN = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-14-qwen-family/evidence/sources"
H100 = "nvidia-h100-sxm-80gb-20260913"

HARDWARE = dict(
    peak_tflops=312, bandwidth_bytes_s=2e12, memory_bytes=80 * 1024**3,
    source="https://example.test/gpu", version="test",
)


def _cfg(folder):
    return json.loads(folder.read_text()) if folder.suffix == ".json" else json.loads((folder / "config.json").read_text())


def test_morphology_tags_k2_and_k3():
    k2 = inspect_model(_cfg(KIMI / "moonshotai--Kimi-K2-Instruct"))
    assert k2["morphology"]["attn"] == ["MLA"]
    assert "MOE" in k2["morphology"]["ffn"]
    assert k2["morphology"]["cost_supported"] is True
    k3 = inspect_model(_cfg(KIMI / "moonshotai--Kimi-K3"))
    assert "KDA" in k3["morphology"]["attn"]
    assert "MLA" in k3["morphology"]["attn"]
    assert "LATENT_MOE" in k3["morphology"]["ffn"]
    assert k3["morphology"]["vision"] == "MoonViT"
    assert k3["morphology"]["quant"] == "mxfp4-group"
    moe = next(t for t in k3["templates"] if t["ffn_kind"] == "MOE")
    assert moe["moe"]["latent_dim"] == 3584
    assert any(n.get("cluster") == "latent" for n in moe["nodes"])
    assert any(n.get("cluster") == "q_path" for n in moe["nodes"])


def test_v4_and_v32_keep_moe_clusters():
    v4 = inspect_model(_cfg(MC / "deepseek-ai--DeepSeek-V4-Flash-Base"))
    assert "CSA" in v4["morphology"]["attn"]
    assert all(t.get("moe") for t in v4["templates"])
    v32 = inspect_model(_cfg(MC / "deepseek-ai--DeepSeek-V3.2-Exp-Base"))
    assert "DSA" in v32["morphology"]["attn"]
    moe = next(t for t in v32["templates"] if t["ffn_kind"] == "MOE")
    assert {n.get("cluster") for n in moe["nodes"]} >= {"router", "routed", "shared", "merge"}


def test_operator_cost_mla_moe_decode():
    k2 = inspect_model(_cfg(KIMI / "moonshotai--Kimi-K2-Instruct"))
    cost = operator_cost(
        k2, phase="decode", batch_size=1, sequence_length=1, context_length=128,
        dtype="bf16", hardware=HARDWARE,
    )
    assert cost["formula_version"] == "operator-cost-2.2.0"
    assert cost["flops"] > 0
    assert any(row["operation"] == "qk_matmul" for row in cost["rows"])
    assert any("routed_expert" in row["operation"] for row in cost["rows"])


def test_k3_huge_prefill_is_modeled_as_float():
    k3 = inspect_model(_cfg(KIMI / "moonshotai--Kimi-K3"))
    cost = operator_cost(
        k3, phase="prefill", batch_size=1, sequence_length=1048576,
        context_length=1048576, dtype="bf16", hardware=HARDWARE,
    )
    assert cost["flops"] > 2 ** 53 - 1
    assert isinstance(cost["flops"], float)
    assert cost["formula_version"] == "operator-cost-2.2.0"


def test_int4_ignore_keeps_attention_full_precision():
    cfg = _cfg(KIMI / "moonshotai--Kimi-K2.5")
    result = inspect_model(cfg)
    mem = memory_components(result, cfg)
    packed = result["totals"]["total"] // 2
    assert mem["weight_bytes"] > packed
    assert "ignore-list" in mem["assumptions"][0] or "ignore" in mem["assumptions"][0].lower()


def test_deepseek_moe_and_vl_clusters():
    moe16 = inspect_model(_cfg(MC / "deepseek-ai--deepseek-moe-16b-base"))
    assert moe16["templates"][0]["ffn_kind"] == "DENSE"
    moe = next(t for t in moe16["templates"] if t["ffn_kind"] == "MOE")
    assert {n.get("cluster") for n in moe["nodes"]} >= {"router", "routed", "shared", "merge", "q_path"}
    v31 = inspect_model(_cfg(MC / "deepseek-ai--DeepSeek-V3.1-Base"))
    moe = next(t for t in v31["templates"] if t["ffn_kind"] == "MOE")
    assert {n.get("cluster") for n in moe["nodes"]} >= {"router", "routed", "shared", "merge"}
    vl = inspect_model(_cfg(MC / "deepseek-ai--deepseek-vl-7b-base"))
    assert vl["morphology"]["vision"] == "SAM_SIGLIP"
    vision = next(b for b in vl["branches"] if b["role"] == "vision")
    assert {n.get("cluster") for n in vision["nodes"]} >= {"patch", "encoder", "projector"}
    k25 = inspect_model(_cfg(KIMI / "moonshotai--Kimi-K2.5"))
    moon = next(b for b in k25["branches"] if b["role"] == "vision")
    assert {n.get("cluster") for n in moon["nodes"]} >= {"patch", "encoder", "projector"}


def _packed(params, bits, group_size, scale_elem, *, zp=False, qzeros=False, g_idx=0):
    packed_n = ((params + group_size - 1) // group_size) * group_size
    groups = (params + group_size - 1) // group_size
    extra = groups * scale_elem
    if zp:
        extra += groups
    if qzeros:
        extra += params * bits // 8 // group_size
    return packed_n * bits // 8 + extra + g_idx


def test_int4_ignore_and_group_scale_closed_form():
    cfg = _cfg(KIMI / "moonshotai--Kimi-K2.5")
    result = inspect_model(cfg)
    ignore = quantization_config(cfg)["ignore"]
    attn = next(row for t in result["templates"] for row in t["modules"] if row["id"] == "q_a_proj")
    assert module_ignored(attn["id"], attn["category"], ignore) is True
    assert module_payload_bytes(attn, cfg, 2) == attn["params"] * 2
    dense = next(row for t in result["templates"] for row in t["modules"] if row["id"] == "gate_proj")
    assert module_payload_bytes(dense, cfg, 2) == dense["params"] * 2
    routed = next(row for t in result["templates"] for row in t["modules"] if row["id"] == "routed_expert.gate_proj")
    assert module_ignored(routed["id"], routed["category"], ignore) is False
    assert module_payload_bytes(routed, cfg, 2) == _packed(routed["params"], 4, 32, 2)
    mem = memory_components(result, cfg)
    assert mem["formula_version"] == "static-memory-components-1.3.0"
    assert result["totals"]["total"] // 2 < mem["weight_bytes"] < result["totals"]["total"] * 2


def test_mxfp4_scale_is_one_byte_per_group():
    cfg = _cfg(KIMI / "moonshotai--Kimi-K3")
    result = inspect_model(cfg)
    routed = next(row for t in result["templates"] for row in t["modules"] if row["id"] == "routed_expert_up_proj")
    assert module_payload_bytes(routed, cfg, 2) == _packed(routed["params"], 4, 32, 1)


def test_awq_payload_includes_scale_and_qzeros():
    cfg = _cfg(QWEN / "Qwen--Qwen3-32B-AWQ")
    result = inspect_model(cfg)
    row = next(item for t in result["templates"] for item in t["modules"] if item["id"] == "q_proj")
    assert module_payload_bytes(row, cfg, 2) == _packed(row["params"], 4, 128, 2, qzeros=True)


def test_gptq_aux_includes_g_idx_and_asymmetric_zp():
    row = dict(id="q_proj", category="attention", params=1024, matrix_params=1024, shape=[32, 32], count=1)
    gptq = {"model_type": "qwen3", "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32}}
    assert module_payload_bytes(row, gptq, 2) == _packed(1024, 4, 32, 2, qzeros=True, g_idx=32 * 4)
    zp = {
        "model_type": "kimi_k2",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"g0": {"weights": {"num_bits": 4, "group_size": 32, "symmetric": False}}},
        },
    }
    ffn = dict(id="routed_expert.gate_proj", category="ffn", params=1024, matrix_params=1024, shape=[32, 32], count=1)
    assert module_payload_bytes(ffn, zp, 2) == _packed(1024, 4, 32, 2, zp=True)


def test_flash_next_stays_rejected_without_qsa_methods():
    cfg = _cfg(QWEN / "Qwen--Qwen3.8-Flash-Next")
    with pytest.raises(ValueError, match="QSA indexer"):
        inspect_model(cfg)


def test_k2_morphology_workload_and_cost_api():
    folder = KIMI / "moonshotai--Kimi-K2-Instruct"
    meta = json.loads((folder / "metadata.json").read_text())
    raw = (folder / "config.json").read_bytes()

    def handler(request):
        if "/api/" in request.url.path:
            return httpx.Response(200, json={"sha": meta["sha"], "safetensors": meta.get("safetensors") or {}})
        return httpx.Response(200, content=raw)

    with TestClient(create_app("sqlite:///:memory:")) as client:
        client.app.state.importer = HuggingFaceImporter(httpx.MockTransport(handler))
        imported = client.post("/api/models/import", json={"repo_id": meta["id"]})
        assert imported.status_code == 200, imported.text
        model = imported.json()
        assert model["workload_supported"] is True
        assert "morphology-workload-1.1.0" in model["support_notes"][0]
        capabilities = client.get("/api/workloads/capabilities", params={"model_id": model["id"]}).json()
        assert capabilities["supported"] is False
        assert any(issue["task"] == "W13a" for issue in capabilities["issues"])
        snapshot = client.post(f"/api/models/{model['id']}/structure").json()
        cost = client.post(
            f"/api/models/{model['id']}/structures/{snapshot['id']}/cost",
            json={"hardware_id": H100, "phase": "decode", "sequence_length": 1, "context_length": 128},
        )
        assert cost.status_code == 200, cost.text
        assert cost.json()["formula_version"] == "operator-cost-2.2.0"


def test_k3_1m_prefill_cost_api_returns_float():
    folder = KIMI / "moonshotai--Kimi-K3"
    meta = json.loads((folder / "metadata.json").read_text())
    raw = (folder / "config.json").read_bytes()

    def handler(request):
        if "/api/" in request.url.path:
            return httpx.Response(200, json={"sha": meta["sha"], "safetensors": meta.get("safetensors") or {}})
        return httpx.Response(200, content=raw)

    with TestClient(create_app("sqlite:///:memory:")) as client:
        client.app.state.importer = HuggingFaceImporter(httpx.MockTransport(handler))
        imported = client.post("/api/models/import", json={"repo_id": meta["id"]})
        assert imported.status_code == 200, imported.text
        model = imported.json()
        snapshot = client.post(f"/api/models/{model['id']}/structure").json()
        cost = client.post(
            f"/api/models/{model['id']}/structures/{snapshot['id']}/cost",
            json={
                "hardware_id": H100, "phase": "prefill",
                "batch_size": 1, "sequence_length": 1048576, "context_length": 1048576,
            },
        )
        assert cost.status_code == 200, cost.text
        payload = cost.json()
        assert payload["formula_version"] == "operator-cost-2.2.0"
        assert payload["flops"] > 2 ** 53 - 1
