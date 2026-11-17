"""Independent published totals and algebraic invariants for component adapters."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from app.engines.components import Ledger
from app.engines.model_structure import breakdown, inspect_model

SOURCES = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-13-model-center/evidence/sources"


def config(name):
    return json.loads((SOURCES / ("deepseek-ai--" + name) / "config.json").read_text())


@pytest.mark.parametrize("name,expected", [
    ("deepseek-moe-16b-base", 16375728128),
    ("DeepSeek-V2.5", 235741434880),
    ("deepseek-llm-67b-base", 67425001472),
    ("deepseek-vl-7b-base", 7343990017),
    ("DeepSeek-Coder-V2-Base", 235741434880),
    ("DeepSeek-Prover-V1.5-Base", 6910365696),
    ("deepseek-math-7b-base", 6910365696),
    ("deepseek-math-7b-rl", 6910365696),
    ("DeepSeek-V3-Base", 671026419200),
    ("DeepSeek-R1", 671026419200),
    ("DeepSeek-V3.1-Base", 671026419200),
    ("DeepSeek-V3.2-Exp-Base", 671877944064),
    ("DeepSeek-V4-Flash-Base", 284332240471),
])
def test_official_model_totals_and_conservation(name, expected):
    cfg = config(name)
    original = deepcopy(cfg)
    a = inspect_model(cfg)
    assert a["totals"]["total"] == expected
    assert sum(breakdown(a).values()) == expected
    assert sum(g["metrics"]["params_total"] for g in a["groups"]) + sum(
        row["params"] for row in a["shared_modules"]) == expected
    assert sum(g["count"] for g in a["groups"]) == a["backbone"]["num_layers"]
    assert cfg == original
    for t in a["templates"] + a["branches"]:
        names = {n["id"] for n in t["nodes"]}
        assert len(names) == len(t["nodes"])
        assert all(e["source"] in names and e["target"] in names for e in t["edges"])


def test_linear_and_gelu_swiglu_small_independent_golden():
    ledger = Ledger("synthetic-test")
    ledger.linear("dense", 3, 5, bias=True, count=7, active_count=2)
    assert sum(r["params"] for r in ledger.rows) == (3*5+5)*7
    assert sum(r["active_params"] for r in ledger.rows) == (3*5+5)*2
    assert sum(r["matrix_params"] for r in ledger.rows) == 3*5*2
    gelu, swiglu = Ledger("synthetic-test"), Ledger("synthetic-test")
    gelu.ffn(3, 5, gated=False, bias=True)
    swiglu.ffn(3, 5, gated=True, bias=True)
    assert sum(r["params"] for r in gelu.rows) == 3*5+5+5*3+3
    assert sum(r["params"] for r in swiglu.rows) == 2*(3*5+5)+5*3+3


def test_moe_frequency_and_topk_change_only_active_weights():
    cfg = config("deepseek-moe-16b-base")
    base = inspect_model(cfg)
    changed = inspect_model({**cfg, "num_experts_per_tok": 2})
    assert changed["totals"]["total"] == base["totals"]["total"]
    assert base["totals"]["active"] - changed["totals"]["active"] == 27*4*3*2048*1408
    sparse = inspect_model({**cfg, "moe_layer_freq": 2})
    assert sum(g["count"] for g in sparse["groups"]
               if next(t for t in sparse["templates"] if t["id"] == g["template_id"])["ffn_kind"] == "MOE") == 13


def test_dsa_indexer_independent_delta():
    v3, dsa = inspect_model(config("DeepSeek-V3-Base")), inspect_model(config("DeepSeek-V3.2-Exp-Base"))
    per_layer = 1536*64*128 + 7168*128 + 2*128 + 7168*64
    assert dsa["totals"]["total"] - v3["totals"]["total"] == 61*per_layer
    assert dsa["serving"]["kv_layout"]["bytes_per_token"] == 61*(512+64+128)*2


def test_v4_mtp_and_precision_not_parameter_identity():
    cfg = config("DeepSeek-V4-Flash-Base")
    a = inspect_model(cfg)
    mtp = inspect_model(cfg, include_mtp=True)
    assert mtp["totals"]["total"] - a["totals"]["total"] == 6610048891
    fp4 = inspect_model({**cfg, "expert_dtype": "fp4"})
    assert fp4["totals"]["total"] == a["totals"]["total"]
    assert a["serving"]["weight_bytes"] - fp4["serving"]["weight_bytes"] == 43*256*3*4096*2048/2
    counts = {}
    for group in a["groups"]:
        t = next(t for t in a["templates"] if t["id"] == group["template_id"])
        counts[t["attn_kind"]] = counts.get(t["attn_kind"], 0) + group["count"]
    assert counts == {"SWA": 2, "CSA": 21, "HCA": 20}


def test_qwen_hybrid_main_and_checkpoint_mtp_golden():
    cfg = json.loads((SOURCES / "Qwen--Qwen3.8-27B/config.json").read_text())
    main = inspect_model(cfg)
    full = inspect_model(cfg, include_mtp=True)
    assert main["totals"]["total"] == 27356728560
    assert full["totals"]["total"] == 27781427952
    assert full["totals"]["mtp"] == 424699392
    assert [g["count"] for g in main["groups"]] == [48, 16]


@pytest.mark.parametrize("patch", [
    {"num_experts_per_tok": 257}, {"head_dim": True}, {"hc_mult": 0},
    {"compress_ratios": [0]*43}, {"num_key_value_heads": 2}, {"o_groups": 7},
    {"expert_dtype": "int1"}, {"hidden_act": "gelu"},
])
def test_invalid_compressed_config_fails_explicitly(patch):
    with pytest.raises(ValueError):
        inspect_model({**config("DeepSeek-V4-Flash-Base"), **patch})


def test_unknown_architecture_no_dense_fallback():
    with pytest.raises(ValueError, match="unsupported"):
        inspect_model({**config("deepseek-math-7b-base"), "hidden_act": "gelu"})
