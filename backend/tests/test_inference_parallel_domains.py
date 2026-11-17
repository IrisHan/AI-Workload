"""Source divisibility is shared by automatic search and UI defaults."""
from dataclasses import replace

from app.engines.inference.search import parallel_domains
from test_inference_partition import mock_ir


def test_domains_dense_moe_and_budget():
    assert parallel_domains(mock_ir(), 8) == {"tp": [1, 2], "ep": [1], "pp": [1, 2]}
    assert parallel_domains(mock_ir(moe=True), 4) == {"tp": [1, 2], "ep": [1, 2, 4], "pp": [1, 2]}
    assert parallel_domains(mock_ir(moe=True), 1) == {"tp": [1], "ep": [1], "pp": [1]}


def test_tensor_axis_divisibility_can_restrict_head_divisors():
    ir = mock_ir()
    first = ir.layers[0]
    weights = tuple(replace(w, shape=(31, 16)) if w.module_id == "up_proj" else w for w in first.weights)
    ir = replace(ir, layers=(replace(first, weights=weights), *ir.layers[1:]))
    assert parallel_domains(ir, 8)["tp"] == [1]
