"""Pure hardware roofline; no workload, database or runtime estimates."""
import math

from app.hardware.schemas import VERIFIED, HardwareRecord, Precision, RooflinePoint, RooflineResult


def require_planner_precision(hardware, precision: str):
    if isinstance(hardware, HardwareRecord) and precision not in hardware.planner_precisions:
        raise ValueError(f"{hardware.name}: {precision} planning requires verified memory and compute data")


def evaluate(hardware: HardwareRecord, precision: Precision, sparse: bool, intensity: float) -> RooflineResult:
    if not math.isfinite(intensity) or not 0 < intensity <= 1e9:
        raise ValueError("Arithmetic intensity must be finite, > 0 and <= 1e9 op/byte")
    for field in ("memory_bytes", "memory_bandwidth_bytes_s"):
        if hardware.field_evidence[field].confidence not in VERIFIED:
            raise ValueError(f"{hardware.name}: unverified {field}; roofline unavailable")
    compute = hardware.compute.get(precision)
    if compute is None or compute.status != "supported":
        raise ValueError(f"{hardware.name}: {precision} unsupported or unverified")
    peak = compute.sparse if sparse else compute.dense
    if peak is None:
        raise ValueError(f"{hardware.name}: {precision} sparse peak unavailable")
    bandwidth = hardware.memory_bandwidth_bytes_s
    ridge = peak * 1e12 / bandwidth
    start = min(0.1, intensity / 10)
    end = max(10000, ridge * 10, intensity * 10)
    positions = {10 ** (math.log10(start) + i / 80 * math.log10(end / start)) for i in range(81)}
    positions.update([ridge, intensity])

    def point(value):
        return RooflinePoint(intensity=value, performance_tops=min(peak, value * bandwidth / 1e12))

    return RooflineResult(
        hardware_id=hardware.id, hardware_version=hardware.version, hardware_sha256=hardware.content_sha256,
        precision=precision, sparse=sparse, unit=compute.unit, peak_tops=peak,
        bandwidth_bytes_s=bandwidth, ridge_point=ridge, operating_point=point(intensity),
        bottleneck="compute" if intensity >= ridge else "memory",
        curve=[point(value) for value in sorted(positions)],
        assumptions=[
            "Theoretical memory roofline, not measured performance or utilization.",
            "P = min(peak operations/s, arithmetic intensity * memory bytes/s); 2 operations per MAC.",
            "SI bandwidth. NVLink, PCIe and NIC are not substituted for HBM/GDDR bandwidth.",
            "Sparse peaks require supported structured sparsity kernels; dense is the default.",
            "INT8 uses TOPS and op/byte, not floating-point TFLOPS.",
        ],
    )
