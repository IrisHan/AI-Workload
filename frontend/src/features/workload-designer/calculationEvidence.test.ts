import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import type { LayerDetail, Plan, Run, Trace } from "../../types/workload";
import { calculationContext, displayFact, groupLayers, layerShards, relevantInputs, validateCalculation } from "./calculationEvidence";

const frozen = JSON.parse(fs.readFileSync(path.resolve(import.meta.dirname,
  "../../../../docs/reports/2026-10-08-calculation-explainability/evidence/live-final-export.json"), "utf8"));
const saved = (): Plan => structuredClone(frozen.plans[0]);
const layer = (index: number): LayerDetail => ({
  layer_index: index, instance_id: `source/${index}`, rank: 0, stage: 0, phase: "prefill", template_id: "dense",
  weight_storage_refs: [`source/${index}/q`],
  local_shapes: [{ storage_id: `source/${index}/q`, module_id: "q", local_shape: [4, 4], count: 1,
    bias: 4, bytes: 40, partition: "column", expert_start: null, expert_stop: null }],
  flops: 32, hbm_traffic_bytes: 40, scope: "representative", operators: [{ id: "q", layer_index: index, seconds: 1 }],
  cache: { total_bytes: 73 }, activation_peak_bytes: 5,
});
describe("semantic layer templates", () => {
  it("groups 28 identical layers by semantics without mutating raw identity or dropping rows", () => {
    const rows = Array.from({ length: 28 }, (_, i) => layer(i)), before = JSON.stringify(rows);
    const groups = groupLayers(rows);
    expect(groups).toHaveLength(1);
    expect(groups[0].layers).toHaveLength(28);
    expect(groups[0].weightBytes).toBe(40);
    expect(groups[0].layers[27]).toBe(rows[27]);
    expect(JSON.stringify(rows)).toBe(before);
    expect(layerShards(rows[0])[0].bytes).toBe(40);
  });
  it.each(["stage", "rank", "phase", "template", "cache", "shape", "expert", "activation", "operator"])(
    "retains genuine %s differences", field => {
      const a = layer(0), b = layer(1);
      if (field === "stage") b.stage = 1;
      if (field === "rank") b.rank = 1;
      if (field === "phase") b.phase = "decode";
      if (field === "template") b.template_id = "sliding";
      if (field === "cache") b.cache.total_bytes++;
      if (field === "shape") layerShards(b)[0].local_shape = [8, 2];
      if (field === "expert") layerShards(b)[0].expert_start = 8;
      if (field === "activation") b.activation_peak_bytes++;
      if (field === "operator") b.operators = [{ id: "q", layer_index: 1, seconds: 2 }];
      expect(groupLayers([a, b])).toHaveLength(2);
    });
  it("does not invent absent shard totals and handles empty results", () => {
    const row = layer(0); row.local_shapes = [null];
    expect(groupLayers([row])[0].weightBytes).toBeNull();
    expect(groupLayers([])).toEqual([]);
  });
});
describe("progressive dependencies", () => {
  it("ITL exposes only service, wait and sample count; complete trace stays untouched", () => {
    const trace = saved().summary.explanation!.traces.itl_ms, original = JSON.stringify(trace);
    expect(relevantInputs(trace).map(f => f.key)).toEqual(["itl_samples", "itl_wave_service_ms", "itl_between_wave_wait_ms"]);
    expect(JSON.stringify(trace)).toBe(original);
    expect(trace.inputs.length).toBeGreaterThan(15);
  });
  it("renames the wave service sum without changing its raw key", () => {
    const trace = saved().summary.explanation!.traces.iteration_time_ms;
    expect(relevantInputs(trace).map(f => f.key)).toEqual(["decode_wall_total_ms", "decode_wave_count"]);
    expect(displayFact(trace.inputs[0]).label).toBe("Sum of Decode-wave Service");
    expect(trace.inputs[0].label).not.toBe("Sum of Decode-wave Service");
  });
  it("keeps shared context once and does not label unrelated envelope fields as dependencies", () => {
    const p = saved();
    expect(calculationContext(p).map(f => f.key)).toEqual(["window_seconds", "gpu_count", "dense_peak", "decode_wave_count", "emitted_tokens"]);
    expect(relevantInputs(p.summary.explanation!.traces.itl_max_ms)).toEqual([]);
    expect(relevantInputs({ ...p.summary.explanation!.traces.kv, id: "kv", inputs: [
      { key: "context", label: "Context", value: 2329, source: "Workload", unit: "tokens" },
      { key: "layer_0", label: "Layer 0", value: 73, source: "Derived", unit: "bytes" },
    ] } as Trace).map(f => f.key)).toEqual(["context"]);
  });
});
describe("calculation invariants", () => {
  it("accepts frozen real TP2 evidence and preserves all source data", () => {
    const p = saved(), before = JSON.stringify(p);
    const result = validateCalculation(p, [], frozen.run as Run);
    expect(result.filter(c => c.status === "fail")).toEqual([]);
    expect(result.filter(c => c.status === "pass").length).toBeGreaterThan(20);
    expect(JSON.stringify(p)).toBe(before);
  });
  it.each(["peak", "free", "iteration", "throughput", "qps", "mfu", "itl", "scope", "nonfinite", "envelope"])(
    "reports a %s inconsistency without correcting the source", field => {
      const p = saved(), raw = p.summary.performance!.metrics;
      if (field === "peak") p.summary.memory.peak_bytes++;
      if (field === "free") p.summary.memory.free_bytes--;
      if (field === "iteration") raw.iteration_time_ms = Number(raw.iteration_time_ms) + 1;
      if (field === "throughput") raw.output_tokens_s = Number(raw.output_tokens_s) + 1;
      if (field === "qps") raw.achieved_qps = Number(raw.achieved_qps) + 1;
      if (field === "mfu") raw.mfu = Number(raw.mfu) + 0.1;
      if (field === "itl") raw.itl_ms = Number(raw.itl_ms) + 1;
      if (field === "scope") p.summary.explanation!.traces.itl_ms.scope.run_id = "different-run";
      if (field === "nonfinite") raw.output_tokens_s = Infinity;
      if (field === "envelope") p.summary.metric_values.output_tokens_s.value = 999;
      const before = JSON.stringify(p);
      expect(validateCalculation(p).some(c => c.status === "fail")).toBe(true);
      expect(JSON.stringify(p)).toBe(before);
    });
  it("does not treat missing or null metrics as a verified zero", () => {
    const p = saved();
    p.summary.performance!.metrics.output_tokens_s = null;
    p.summary.metric_values.output_tokens_s.value = null;
    const checks = validateCalculation(p);
    expect(checks.find(c => c.name === "Output Throughput")?.status).toBe("unverifiable");
  });
  it("uses floating point tolerance but exact memory bytes", () => {
    const p = saved();
    p.summary.performance!.metrics.output_tokens_s = Number(p.summary.performance!.metrics.output_tokens_s) + 1e-10;
    expect(validateCalculation(p).find(c => c.name === "Output Throughput")?.status).toBe("pass");
    p.summary.memory.peak_bytes++;
    expect(validateCalculation(p).find(c => c.name === "Admission · Peak components")?.status).toBe("fail");
  });
  it("detects divergent dependency inputs even when the displayed metric matches raw metrics", () => {
    const p = saved();
    p.summary.explanation!.traces.output_tokens_s.inputs.find(f => f.key === "emitted_tokens")!.value = 999;
    expect(validateCalculation(p).some(c => c.name.includes("binding") && c.status === "fail")).toBe(true);
  });
  it("rejects non-finite metrics even if another operand is missing", () => {
    const p = saved();
    p.summary.performance!.metrics.output_tokens_s = Infinity;
    delete p.summary.performance!.metrics.window_seconds;
    expect(validateCalculation(p).find(c => c.name === "Output Throughput")?.status).toBe("fail");
  });
  it("keeps historical missing evidence unverified", () => {
    const p = saved(); delete p.summary.explanation; delete p.summary.memory.explanation;
    expect(validateCalculation(p).find(c => c.name === "Decode-wave service total")?.status).toBe("unverifiable");
    expect(validateCalculation(p).filter(c => c.name.startsWith("Scope")).every(c => c.status !== "pass")).toBe(true);
  });
  it("preserves ITL's request-sample denominator and the separate wave-service denominator", () => {
    const p = saved(), raw = p.summary.performance!.metrics;
    const inputs = p.summary.explanation!.traces.itl_ms.inputs;
    const service = Number(inputs.find(f => f.key === "itl_wave_service_ms")!.value);
    const wait = Number(inputs.find(f => f.key === "itl_between_wave_wait_ms")!.value);
    expect(Number(raw.itl_ms)).toBeCloseTo(service + wait, 8);
    expect(Number(raw.itl_ms)).toBeGreaterThan(Number(raw.iteration_time_ms));
    expect(validateCalculation(p).find(c => c.name === "Decode-wave service total")?.status).toBe("pass");
  });
});
