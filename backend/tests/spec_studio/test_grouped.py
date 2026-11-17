import json

import pytest
from app.lib.spec_studio.inspect import inspect_config, inspect_config_v2
from app.lib.spec_studio.matcher import match_config
from app.lib.spec_studio.validation import ensure_safe_numbers


@pytest.mark.parametrize(
    "fixture_name,counts,total",
    [
        ("qwen_cfg", [28], 1543714304),
        ("v3_cfg", [3, 58], 671026419200),
        ("qwen1_72b_cfg", [80], 72285954048),
        ("mixtral_cfg", [32], 46702792704),
    ],
)
def test_grouped_preserves_independent_goldens(request, fixture_name, counts, total):
    cfg = request.getfixturevalue(fixture_name)
    result = inspect_config_v2(cfg)
    assert [group["count"] for group in result["groups"]] == counts
    assert result["totals"]["total"] == total
    assert result["totals"]["total"] == inspect_config(cfg).arch.totals.total
    assert "layers" not in result
    for group in result["groups"]:
        template = next(row for row in result["templates"] if row["id"] == group["template_id"])
        assert sum(row["params"] for row in template["modules"]) == template["metrics"]["params_total"]
        assert (
            sum(row["active_params"] for row in template["modules"]) == template["metrics"]["params_active"]
        )
        assert template["metrics"]["params_total"] * group["count"] == group["metrics"]["params_total"]
        ids = {node["id"] for node in template["nodes"]}
        assert all(edge["source"] in ids and edge["target"] in ids for edge in template["edges"])
        assert {"attn_add", "ffn_add", "softmax", "qk_matmul"} <= ids
    assert (
        sum(row["params"] for row in result["shared_modules"])
        + sum(group["metrics"]["params_total"] for group in result["groups"])
        == total
    )
    assert (
        sum(row["active_params"] for row in result["shared_modules"])
        + sum(group["metrics"]["params_active"] for group in result["groups"])
        == result["totals"]["active"]
    )


def test_grouped_512_layers_constant_templates(qwen_cfg, monkeypatch):
    qwen_cfg["num_hidden_layers"] = 512
    monkeypatch.setattr(
        "app.lib.spec_studio.graph.build_graph", lambda *args: pytest.fail("expanded graph called")
    )
    result = inspect_config_v2(qwen_cfg)
    assert len(result["groups"]) == 1
    assert result["groups"][0]["count"] == 512
    assert len(match_config(qwen_cfg, compact=True)["layers"]) == 1
    assert len(json.dumps(result).encode()) < 100000
    assert len(result["templates"][0]["nodes"]) < 80


def test_grouped_alternating_attention_preserves_order(qwen_cfg):
    qwen_cfg.update(
        num_hidden_layers=4,
        use_sliding_window=True,
        sliding_window=128,
        layer_types=["full_attention", "sliding_attention"] * 2,
    )
    result = inspect_config_v2(qwen_cfg)
    assert [group["count"] for group in result["groups"]] == [2, 2]
    assert [row["group_id"] for row in result["execution"]] == ["g1", "g2", "g1", "g2"]
    assert result["groups"][0]["ranges"] == [[0, 1], [2, 3]]
    assert result["templates"][0]["signature"] != result["templates"][1]["signature"]


def test_grouped_dense_prefix_has_no_experts(v3_cfg):
    result = inspect_config_v2(v3_cfg)
    assert not any("expert" in row["id"] for row in result["templates"][0]["modules"])
    routed = [row for row in result["templates"][1]["modules"] if row["id"].startswith("routed_expert.")]
    assert len(routed) == 3
    assert all(row["count"] == 256 for row in routed)
    assert all(row["active_params"] * 32 == row["params"] for row in routed)


def test_grouped_tied_head_and_final_norm(qwen_cfg):
    rows = {row["id"]: row for row in inspect_config_v2(qwen_cfg)["shared_modules"]}
    assert rows["lm_head"]["params"] == 0
    assert rows["lm_head"]["matrix_params"] == 233373696
    assert rows["lm_head"]["shared_with"] == "embed"
    assert rows["final_norm"]["params"] == 1536


def test_grouped_mtp_separate_and_active_topk(v3_cfg):
    off = inspect_config_v2(v3_cfg)
    on = inspect_config_v2(v3_cfg, include_mtp=True)
    mtp = next(row for row in on["shared_modules"] if row["id"] == "mtp")
    assert mtp["params"] == 11610061056
    assert mtp["active_params"] < mtp["params"]
    assert on["totals"]["total"] - off["totals"]["total"] == mtp["params"]
    assert [group["count"] for group in on["groups"]] == [3, 58]


@pytest.mark.parametrize(
    "change,error",
    [
        ({"num_hidden_layers": 0}, "integer"),
        ({"num_hidden_layers": 4097}, "integer"),
        ({"num_attention_heads": 0}, "integer"),
        ({"hidden_size": 1.5}, "integer"),
        ({"hidden_size": True}, "integer"),
        ({"vocab_size": -1}, "integer"),
        ({"num_key_value_heads": 5}, "divisible"),
        ({"layer_types": ["full_attention"]}, "exactly"),
        ({"index_n_heads": 4}, "indexer"),
        ({"vision_config": {}}, "unsupported"),
        ({"compress_ratios": []}, "unsupported"),
        ({"architectures": ["LlamaForCausalLM"]}, "does not match"),
        ({"hidden_act": "not-an-activation"}, "unsupported"),
        ({"torch_dtype": "complex128"}, "unsupported"),
        ({"attention_bias": "false"}, "boolean"),
        ({"layer_norm_epsilon": 1e-6}, "unsupported"),
    ],
)
def test_grouped_invalid_configs_fail_closed(qwen_cfg, change, error):
    qwen_cfg.update(change)
    with pytest.raises(ValueError, match=error):
        inspect_config_v2(qwen_cfg)


@pytest.mark.parametrize("topk", [0, -1, 257, 1.2, True])
def test_moe_topk_validation(v3_cfg, topk):
    v3_cfg["num_experts_per_tok"] = topk
    with pytest.raises(ValueError, match="integer"):
        inspect_config_v2(v3_cfg)


def test_safe_integer_guard():
    with pytest.raises(ValueError, match="safe integer"):
        ensure_safe_numbers({"params": 2**53})
    ensure_safe_numbers({"flops": float(2**60)})


def test_kv_precision_explicit(qwen_cfg):
    result = inspect_config_v2(qwen_cfg, kv_dtype="fp32")
    assert result["serving"]["kv_layout"]["bytes_per_token"] == 57344
    assert result["formula_version"] == "2.0.0"


def test_signature_independent_of_repeated_count(qwen_cfg):
    original = inspect_config_v2(qwen_cfg)
    qwen_cfg["num_hidden_layers"] = 512
    large = inspect_config_v2(qwen_cfg)
    assert original["templates"][0]["signature"] == large["templates"][0]["signature"]


def test_qk_norm_inventory_is_counted(qwen_cfg):
    qwen_cfg["qk_norm"] = True
    result = inspect_config_v2(qwen_cfg)
    rows = {row["id"]: row for row in result["templates"][0]["modules"]}
    assert rows["q_norm"]["params"] == rows["k_norm"]["params"] == 128
    assert rows["q_norm"]["matrix_params"] == 0
    assert result["totals"]["total"] == 1543714304 + 28 * 256


def test_mtp_matrix_inventory_conservation(v3_cfg):
    result = inspect_config_v2(v3_cfg, include_mtp=True)
    matrix = sum(group["metrics"]["matrix_params"] for group in result["groups"])
    matrix += sum(row["matrix_params"] for row in result["shared_modules"])
    assert matrix * 2 == result["serving"]["flop_gemm_decode"]
