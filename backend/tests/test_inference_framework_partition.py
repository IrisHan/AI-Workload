"""框架布局独立金标：mock 数据，无真实 GPU 性能断言。"""
import pytest
from app.engines.inference.cache import cache_allocation, request_quotas
from app.engines.inference.ir import Candidate, PlanningError
from app.engines.inference.partition import (
    default_boundaries,
    groups,
    make_candidate,
    rank_weights,
    ranks,
    validate_candidate,
)
from test_inference_partition import design, mock_ir


@pytest.mark.parametrize("layout,expected", [
    ("vllm-v0.10.2", (9, 19)), ("sglang-v0.5.3", (9, 18)),
])
def test_default_boundaries_28_layers(layout, expected):
    assert default_boundaries(28, 3, layout) == expected
    assert default_boundaries(28, 1, layout) == ()


@pytest.mark.parametrize("layout,tp,ep,dp,enabled,etp,attention,replicas", [
    ("vllm-v0.10.2", 2, 4, 2, True, 1, 2, 1),
    ("vllm-v0.10.2", 2, 1, 2, False, 4, 2, 1),
    ("sglang-v0.5.3", 4, 2, 2, False, 2, 1, 2),
])
def test_framework_device_expert_and_request_conservation(layout, tp, ep, dp, enabled, etp, attention, replicas):
    ir = mock_ir(moe=True)
    inp = design(parallelism={"layout": layout, "mode": "manual", "expert_parallel": enabled,
                             "manual_candidate": {"tp": tp, "pp": 1, "ep": ep, "dp": dp, "boundaries": []}})
    c = make_candidate(ir, inp, inp.parallelism.manual_candidate)
    validate_candidate(ir, c, inp)
    assert c.gpu_count == tp * dp
    assert (c.expert_tp, c.attention_groups, c.independent_replicas) == (etp, attention, replicas)
    placements = ranks(c, 8)
    assert len({r.rank for r in placements}) == tp * dp
    assert sum(sum(row) for row in request_quotas(13, c)) == 13
    expected = 2 * 4 * 3 * 32 * 16 * 2 * replicas  # layers × experts × matrices × shape × BF16 × replicas
    observed = sum(w.bytes for rank in placements for w in rank_weights(ir, c, rank, "bf16")
                   if w.expert_start is not None)
    assert observed == expected
    assert all(len(g["ranks"]) == etp for g in groups(c, placements) if g["kind"] == "etp")
    assert all(len(g["ranks"]) == ep for g in groups(c, placements) if g["kind"] == "ep")


def test_kv_replication_keeps_one_head_and_weight_per_lane():
    ir = mock_ir()
    c = Candidate(4, 1, 1, 1, (), "sglang-v0.5.3")
    validate_candidate(ir, c, design(parallelism={"layout": c.layout}))
    kv = cache_allocation(ir.layers[0], [16], 4, "bf16")
    assert kv["payload_bytes"] == 16 * 2 * 1 * 4 * 2
    for rank in ranks(c, 8):
        shard = next(w for w in rank_weights(ir, c, rank, "bf16") if w.module_id == "k_proj")
        assert shard.local_shape == (4, 16)


@pytest.mark.parametrize("layout,tp,ep,dp,enabled,code", [
    ("vllm-v0.10.2", 2, 2, 2, True, "FRAMEWORK_EP"),
    ("sglang-v0.5.3", 2, 4, 1, False, "FRAMEWORK_EP"),
])
def test_invalid_framework_ep_is_explicit(layout, tp, ep, dp, enabled, code):
    inp = design(parallelism={"layout": layout, "expert_parallel": enabled})
    with pytest.raises(PlanningError) as exc:
        validate_candidate(mock_ir(moe=True), Candidate(tp, 1, ep, dp, (), layout, True), inp)
    assert exc.value.code == code
