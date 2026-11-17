import json
from pathlib import Path

import pytest
from app.engines.model_memory import memory_components
from app.engines.model_structure import inspect_model

SOURCES = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-13-model-center/evidence/sources"


@pytest.fixture
def v4():
    cfg = json.loads((SOURCES / "deepseek-ai--DeepSeek-V4-Flash-Base/config.json").read_text())
    return cfg, inspect_model(cfg)


def test_compressed_cache_independent_at_boundaries(v4):
    cfg, a = v4
    for s in (1, 3, 4, 127, 128, 4096, 1048576):
        result = memory_components(a, cfg, context_length=s)
        window = 43*min(s, 128)*512*2
        pools = (21*(s//4)*(512+128)+20*(s//128)*512)*2
        state = 21*2*2*4*2*(512+128)*4 + 20*2*128*512*4
        assert result["kv_cache_bytes"] == window+pools+state
        assert result["weight_bytes"] == 284332240471*2
        assert result["integer_buffer_bytes"] == 3 * 129280 * 6 * 4
        assert result["static_total_bytes"] == result["weight_bytes"] + window+pools+state + result["integer_buffer_bytes"]
        assert result["activation_bytes"] is None and result["peak_memory_bytes"] is None


def test_batch_and_precision_scaling_and_training(v4):
    cfg, a = v4
    base = memory_components(a, cfg)
    double = memory_components(a, cfg, batch_size=2)
    assert double["kv_cache_bytes"] == 2*base["kv_cache_bytes"]
    assert double["weight_bytes"] == base["weight_bytes"]
    fp32 = memory_components(a, cfg, dtype="fp32")
    assert fp32["weight_bytes"] == 2*base["weight_bytes"]
    assert fp32["kv_cache_bytes"] < 2*base["kv_cache_bytes"]  # staging already FP32
    train = memory_components(a, cfg, include_training_state=True)
    assert train["kv_cache_bytes"] == 0
    assert train["gradient_bytes"] == train["weight_bytes"]
    assert train["optimizer_bytes"] == 284332240471*12


@pytest.mark.parametrize("overrides", [
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 4097},
    {"context_length": 1048577}, {"context_length": -1}, {"dtype": "int4"},
])
def test_invalid_memory_request(v4, overrides):
    cfg, a = v4
    with pytest.raises(ValueError):
        memory_components(a, cfg, **overrides)


def test_hybrid_recurrent_state_constant_in_context():
    cfg = json.loads((SOURCES / "Qwen--Qwen3.8-27B/config.json").read_text())
    a = inspect_model(cfg)
    p = memory_components(a, cfg, context_length=128)
    q = memory_components(a, cfg, context_length=256)
    assert q["kv_cache_bytes"] - p["kv_cache_bytes"] == 16*128*2*4*256*2
    assert p["cache_components"][0]["recurrent_state_bytes"] == 48*128*128*4
