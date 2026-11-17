"""Metric provenance and explicit scenario sensitivity, never measured data."""
import copy
import time
from dataclasses import replace

from .scheduler import simulate

UNITS = {
    "ttft_ms": "ms", "ttft_max_ms": "ms", "itl_ms": "ms", "itl_max_ms": "ms",
    "service_ttft_ms": "ms", "end_to_end_ms": "ms",
    "output_tokens_s": "tokens/s", "tokens_s_per_gpu": "tokens/s/GPU",
    "prefill_tokens_s": "tokens/s", "achieved_qps": "requests/s", "offered_qps": "requests/s",
    "completion_seconds": "s", "modeled_busy_fraction": "fraction",
    "mfu": "fraction", "iteration_time_ms": "ms", "prefill_tokens_s_per_gpu": "tokens/s/GPU",
    "prefill_busy_fraction": "fraction", "decode_busy_fraction": "fraction",
    "gpu_activity_fraction": "fraction",
    "cost_per_million_output_tokens": "CNY/1M output tokens",
    "energy_per_million_output_tokens": "J/1M output tokens",
}


def metric_values(simulation, scenario="baseline"):
    values = {}
    for name, unit in UNITS.items():
        value = simulation["metrics"].get(name)
        reason = None
        if value is None:
            reason = ("Price scope/date or power operating scenario unavailable" if name.startswith(("cost_", "energy_"))
                      else "No complete samples, O=1, or required timing unavailable")
        values[name] = {"value": value, "unit": unit, "kind": "modeled", "scenario_id": scenario,
                        "confidence_basis": "Frozen analytical runtime assumptions; no GPU measurements",
                        "unavailable_reason": reason,
                        "scope": "per_gpu" if name.endswith("_per_gpu") else "cluster"}
    return values


def sensitivity(sources, candidate, design, deadline, cancelled):
    rows = []
    for name, compute, hbm in (("low_efficiency", .3, .5), ("high_efficiency", .7, .85)):
        if cancelled() or time.monotonic() >= deadline:
            rows.append({"scenario_id": name, "status": "not_evaluated", "reason": "diagnostic_budget"})
            continue
        assumptions = copy.deepcopy(sources.assumptions)
        assumptions["compute_efficiency"]["effective"] = compute
        assumptions["hbm_efficiency"]["effective"] = hbm
        simulation = simulate(replace(sources, assumptions=assumptions), candidate, design, 10000,
                              cancelled=lambda: cancelled() or time.monotonic() >= deadline)
        rows.append({"scenario_id": name, "compute_efficiency": compute, "hbm_efficiency": hbm,
                     "status": simulation["status"], "metrics": metric_values(simulation, name),
                     "meaning": "Alternative assumptions, not a probability or confidence interval"})
    return rows
