"""Matrix/scalar scenario separation; unknown timing propagates."""


def operator_time(flops, traffic_bytes, hardware, assumptions, precision, *, scalar=False):
    key = "fp32" if scalar else precision
    compute = hardware.get("compute", {}).get(key, {})
    peak = compute.get("dense") if compute.get("status") == "supported" else None
    override = assumptions.get("hbm_bandwidth_gb_s", {}).get("effective", 0)
    bw = override * 1e9 if override else hardware.get("memory_bandwidth_bytes_s")
    if not peak or not bw:
        return {"seconds": None, "lower_bound_seconds": None,
                "unknown": f"Verified {'scalar FP32' if scalar else key} throughput required"}
    a = {key: value["effective"] for key, value in assumptions.items()}
    compute_time = flops / (peak * 1e12 * a["compute_efficiency"])
    hbm_time = traffic_bytes / (bw * a["hbm_efficiency"])
    return {"seconds": max(compute_time, hbm_time) + a["launch_time_us"] / 1e6,
            "compute_seconds": compute_time, "hbm_seconds": hbm_time,
            "launch_seconds": a["launch_time_us"] / 1e6,
            "peak_tflops": peak, "hbm_bandwidth_bytes_s": bw,
            "compute_efficiency": a["compute_efficiency"], "hbm_efficiency": a["hbm_efficiency"],
            "lower_bound_seconds": max(flops / (peak * 1e12), traffic_bytes / bw),
            "bottleneck": "compute" if compute_time >= hbm_time else "hbm",
            "unknown": None, "profile": "scalar-equivalent-v1" if scalar else "matrix-roofline-v1"}
