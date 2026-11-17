"""SI-unit timing goldens independent of the implementation."""
import pytest
from app.engines.inference.roofline import operator_time
from test_inference_execution import mock_sources


def test_full_kv_read_si_units():
    source = mock_sources()
    hardware = {**source.hardware, "memory_bandwidth_bytes_s": 2000 * 10**9}
    result = operator_time(0, 67108864, hardware, source.assumptions, "bf16")
    assert result["hbm_seconds"] * 1000 == pytest.approx(0.047934902857)
    assert result["seconds"] * 1000 == pytest.approx(0.052934902857)
    assert result["compute_seconds"] == 0
    assert result["bottleneck"] == "hbm"


def test_compute_bound_uses_verified_dense_and_efficiency():
    source = mock_sources()
    result = operator_time(10**12, 0, source.hardware, source.assumptions, "bf16")
    assert result["compute_seconds"] == pytest.approx(.02)
    assert result["seconds"] == pytest.approx(.020005)
    assert result["peak_tflops"] == 100
    assert result["bottleneck"] == "compute"
