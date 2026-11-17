"""Independent small mock goldens; these are not measured GPU results."""
from dataclasses import replace

import pytest
from app.engines.inference.ir import Candidate, Layer, ModelIR, PlanningError, Weight
from app.engines.inference.normalize import normalize, weight_from_module
from app.engines.inference.partition import groups, rank_weights, ranks, shard_weight, validate_candidate
from app.workload.schemas import DesignInput
from pydantic import ValidationError


def mock_ir(layers=2, moe=False, tied=True):
    def weight(name, shape, rule, count=1, routed=False):
        return Weight(name, name, shape, count, 0, "bf16", rule, routed)

    body = (
        weight("q_proj", (16, 16), "column"), weight("k_proj", (8, 16), "column"),
        weight("v_proj", (8, 16), "column"), weight("o_proj", (16, 16), "row"),
        weight("attn_norm", (16,), "replicate"),
        weight("ffn_norm", (16,), "replicate"),
    )
    prefix = "routed_expert." if moe else ""
    body += tuple(weight(prefix + name, shape, rule, 4 if moe else 1, moe) for name, shape, rule in [
        ("gate_proj", (32, 16), "column"), ("up_proj", (32, 16), "column"),
        ("down_proj", (16, 32), "row")])
    layer_list = tuple(Layer(
        i, f"mock/{i}", "mock-template", "GQA", "MOE" if moe else "DENSE",
        16, 4, 2, 4, 32, 4 if moe else 0, 2 if moe else 0, None,
        tuple(replace(w, storage_id=f"mock/{i}/{w.module_id}") for w in body), (),
    ) for i in range(layers))
    shared = (weight("embed", (32, 16), "vocab"), weight("final_norm", (16,), "replicate"),
              replace(weight("lm_head", (32, 16), "vocab"), alias_of="embed" if tied else None))
    logical = sum(w.numel for layer in layer_list for w in layer.weights)
    logical += sum(w.numel for w in shared if not w.alias_of)
    return ModelIR("mock-model", "mock-snapshot", "mock-revision", "0" * 64, "mock", "mock",
                   layer_list, shared, 32, 16, 65536, logical)


def design(**changes):
    return DesignInput.model_validate({"model_id": "mock-model", "hardware_id": "mock-gpu", **changes})


def test_gqa_tp_norm_goldens():
    ir = mock_ir()
    c = Candidate(2, 1, 1, 1, ())
    attention = ir.layers[0].weights[:4]
    assert sum(w.numel for w in attention) == 768
    assert sum(shard_weight(w, c, 0, "bf16").bytes for w in attention) == 768
    assert shard_weight(ir.layers[0].weights[4], c, 0, "bf16").bytes == 32
    assert shard_weight(attention[1], c, 0, "bf16").local_shape == (4, 16)


def test_g04_orthogonal_experts():
    c = Candidate(2, 1, 2, 1, ())
    weights = [Weight(name, name, shape, 4, 0, "bf16", rule, True) for name, shape, rule in [
        ("gate", (16, 8), "column"), ("up", (16, 8), "column"), ("down", (8, 16), "row")]]
    assert sum(shard_weight(w, c, 0, "bf16").bytes for w in weights) == 768
    assert shard_weight(weights[0], c, 1, "bf16").expert_start == 2
    assert shard_weight(weights[0], c, 1, "bf16").expert_stop == 4


def test_g12_shared_storage_one_or_two_physical_placements():
    ir = replace(mock_ir(), shared=(
        Weight("embed", "embed", (10, 10), 1, 0, "bf16", "vocab", False),
        Weight("head", "lm_head", (10, 10), 1, 0, "bf16", "vocab", False, "embed"),
    ))
    for pp, expected in [(1, 200), (2, 400)]:
        c = Candidate(1, pp, 1, 1, (1,) if pp == 2 else ())
        total = sum(s.bytes for rank in ranks(c, 8) for s in rank_weights(ir, c, rank, "bf16")
                    if s.storage_id == "embed")
        assert total == expected


def test_rank_encoding_and_groups():
    c = Candidate(4, 2, 4, 2, (1,))
    placements = ranks(c, 8)
    assert len(placements) == 64
    assert len({(r.node, r.gpu) for r in placements}) == 64
    for r in placements:
        assert r.rank == (((r.d * 2 + r.p) * 4 + r.e) * 4 + r.t)
    assert all(len(g["ranks"]) == 4 for g in groups(c, placements) if g["kind"] == "ep")


@pytest.mark.parametrize("candidate,code", [
    (Candidate(4, 1, 1, 1, ()), "HEAD_DIVISIBILITY"),
    (Candidate(1, 1, 2, 1, ()), "DENSE_EP"),
    (Candidate(1, 2, 1, 1, (2,)), "INVALID_BOUNDARIES"),
    (Candidate(2, 2, 2, 2, (1,)), "GPU_BUDGET"),
])
def test_candidate_rejection(candidate, code):
    with pytest.raises(PlanningError) as exc:
        validate_candidate(mock_ir(), candidate, design())
    assert exc.value.code == code


def test_merged_bias_count_not_multiplied_twice():
    row = {"id": "routed_expert.gate_proj", "shape": [16, 8], "count": 4,
           "params": 576, "dtype": "bf16"}
    w = weight_from_module(row, "mock")
    assert w.numel == 576
    s = shard_weight(w, Candidate(2, 1, 2, 1, ()), 0, "bf16")
    assert s.bytes == 288
    assert s.bias == 8


@pytest.mark.parametrize("value", [True, 1.5, "8", 0, -1, 4097])
def test_strict_concurrency(value):
    with pytest.raises(ValidationError):
        design(load={"concurrency": value})


@pytest.mark.parametrize("changes", [
    {"load": {"request_rate": float("nan")}},
    {"resources": {"node_count": 256, "gpus_per_node": 2}},
    {"mode": "offline"},
    {"overrides": [{"path": "model.hidden_size", "value": 16, "reason": "invalid"}]},
    {"overrides": [{"path": "runtime_fixed_bytes", "value": 1.5, "reason": "invalid"}]},
    {"runtime": {"continuous_batching": False}},
    {"invented": 1},
])
def test_input_rejection(changes):
    with pytest.raises(ValidationError):
        design(**changes)


def test_current_source_qwen_parameter_conservation():
    from app.application.planner_service import DATA
    from app.integrations.huggingface import normalize_model
    from app.integrations.spec_studio import structure

    model = normalize_model("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323",
                            (DATA / "qwen-1.5b-config.json").read_bytes(), "2026-10-06")
    analysis = structure(model)["analysis"]
    snapshot = {"id": "mock-snapshot", "model_id": model.id, "model_version": model.version,
                "engine_version": analysis["engine_version"], "formula_version": analysis["formula_version"],
                "analysis": analysis}
    ir = normalize(model.model_dump(), snapshot)
    assert ir.logical_numel == 1543714304
    assert len(ir.layers) == 28
    c = Candidate(2, 2, 1, 1, (14,))
    validate_candidate(ir, c, design())
    assert ir.layers[0].weights[1].bias == 1536
    assert len(ir.layers[0].operators) > len(ir.layers[0].weights)
