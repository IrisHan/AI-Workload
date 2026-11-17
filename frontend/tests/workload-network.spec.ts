import { test, expect, type APIRequestContext } from "@playwright/test";
import { initialInput } from "../src/features/workload-designer/state";

async function prepare(request: APIRequestContext) {
  const models = await (await request.get("/api/models")).json();
  const input = initialInput(models.find((m: { repo_id: string }) => m.repo_id === "Qwen/Qwen2.5-0.5B").id,
    "nvidia-h100-sxm-80gb-20260913");
  input.mode = "offline";
  Object.assign(input.load, { prompt_tokens: 5, output_tokens: 3, concurrency: 2, request_count: 2, request_rate: null, arrival: null });
  input.runtime.max_batch_sequences = 2;
  Object.assign(input.resources, { node_count: 4, gpus_per_node: 2 });
  input.parallelism.mode = "manual";
  input.parallelism.manual_candidate = { tp: 2, pp: 2, ep: 1, dp: 2, boundaries: [12] };
  const created = await request.post("/api/workloads/designs", {
    data: { name: `Network acceptance ${Date.now()}`, input }, headers: { "Idempotency-Key": crypto.randomUUID() },
  });
  expect(created.status()).toBe(201);
  const design = await created.json();
  const submitted = await request.post("/api/workloads/runs", {
    data: { design_revision_id: design.revision.id, search: { top_k: 1, max_candidates: 1, max_seconds: 5, event_budget: 5000 } },
    headers: { "Idempotency-Key": crypto.randomUUID() },
  });
  expect(submitted.status()).toBe(202);
  const run = await submitted.json();
  await expect.poll(async () => (await (await request.get(`/api/workloads/runs/${run.id}`)).json()).status,
    { timeout: 20000 }).toBe("completed");
  const plans = await (await request.get(`/api/workloads/runs/${run.id}/plans`)).json();
  return { design, run, plans, plan: plans.items[0] };
}

test("network: one DP, physical nodes, link details, filters and selection", async ({ page, request }, info) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  const { design, run, plan } = await prepare(request);
  const path = `/api/workloads/runs/${run.id}/plans/${plan.id}`;
  const network = await (await request.get(path + "/graph?view=communication")).json();
  expect(network.nodes.filter((n: { category: string }) => n.category === "gpu")).toHaveLength(4);
  expect(network.edges.find((e: { kind: string }) => e.kind === "pp").medium).toContain("NIC");
  await page.goto(`/workloads/inference/${design.id}?run=${run.id}&view=placement`);
  await expect(page.locator(".wc-gpus button")).toHaveCount(4);
  await expect(page.locator(".wc-topology-scope")).toContainText("2 replicas");
  expect(await page.locator(".wc-gpus button").evaluateAll(nodes =>
    nodes.every(n => n.scrollWidth <= n.clientWidth && n.scrollHeight <= n.clientHeight))).toBe(true);
  await expect(page.locator(".wc-source-foot, #wc-evaluate-reason")).toHaveCount(0);
  await page.getByRole("button", { name: "Communication", exact: true }).click();
  await expect(page.locator(".wc-network-machine")).toHaveCount(2);
  await expect(page.locator(".wc-network-replicas")).toHaveCount(1);
  await page.getByLabel("Communication type", { exact: true }).selectOption("pp");
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);
  await page.locator(".wc-network-edge-label").first().click();
  await expect(page.getByLabel("Link details", { exact: true })).toContainText("Inter-node");
  await expect(page.getByLabel("Link details", { exact: true })).toContainText("50 GB/s");
  await expect(page.getByLabel("Link details", { exact: true })).toContainText("100 GB/s shared");
  await page.getByLabel("Phase", { exact: true }).selectOption("decode");
  await expect(page.getByLabel("Link details", { exact: true })).not.toContainText("No saved sample");
  await page.getByRole("button", { name: "Close link details" }).click();
  await page.locator('.react-flow__node[data-id="rank-2"]').click();
  await expect(page.getByLabel("GPU", { exact: true })).toHaveValue("2");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.locator(".wc-workbench").screenshot({ path: `${process.env.PLANNER_E2E_EVIDENCE ?? "/tmp"}/network-${info.project.name}.png` });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(errors).toEqual([]);
});

test("mock layout boundary: 128 cards, 16 nodes, second rank page and folded DP", async ({ page, request }, info) => {
  const { design, run, plans, plan } = await prepare(request);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1920, height: 1000 });
  const placement = Array.from({ length: 256 }, (_, rank) => ({
    rank, node: Math.floor(rank / 8), gpu: rank % 8, d: Math.floor(rank / 128),
    p: Math.floor(rank % 128 / 8), t: rank % 8, e: 0,
  }));
  plan.summary.placement = placement;
  plan.summary.candidate = { tp: 8, pp: 16, ep: 1, dp: 2, boundaries: Array.from({ length: 15 }, (_, i) => i + 1) };
  plan.summary.gpu_count = 256;
  delete plan.summary.explanation;
  delete plan.summary.memory.explanation;
  delete plan.summary.performance;
  plan.summary.metric_values = {};
  const rankTemplate = (await (await request.get(`/api/workloads/runs/${run.id}/plans/${plan.id}/ranks`)).json()).items[0];
  delete rankTemplate.explanation;
  await page.route(`**/api/workloads/runs/${run.id}/plans?*`, route => route.fulfill({ json: plans }));
  await page.route(`**/api/workloads/runs/${run.id}/plans/${plan.id}/ranks?*`, route => {
    const offset = Number(new URL(route.request().url()).searchParams.get("offset"));
    return route.fulfill({ json: { items: placement.slice(offset, offset + 100).map(r => ({ ...rankTemplate, ...r })),
      next_offset: offset + 100 < 256 ? offset + 100 : null, total: 256, result_hash: plan.result_hash } });
  });
  const nodes = placement.filter(r => r.d === 0).map(r => ({ ...r, id: `rank-${r.rank}`,
    label: `GPU ${r.gpu}`, category: "gpu", rank_refs: [r.rank] }));
  const edges = nodes.flatMap(n => [
    { id: `tp-${n.rank}`, source: n.id, target: `rank-${n.rank - n.gpu + (n.gpu + 1) % 8}`,
      kind: "tp", rank_refs: [n.rank, n.rank - n.gpu + (n.gpu + 1) % 8],
      distance: "intra_node", bandwidth_bytes_s: 450e9 },
    ...(n.p < 15 ? [{ id: `pp-${n.rank}`, source: n.id, target: `rank-${n.rank + 8}`,
      kind: "pp", rank_refs: [n.rank, n.rank + 8], distance: "inter_node", bandwidth_bytes_s: 50e9 }] : []),
  ]);
  await page.route(`**/api/workloads/runs/${run.id}/plans/${plan.id}/graph?*`, route => route.fulfill({ json: {
    nodes, edges, plan_id: plan.id, result_hash: plan.result_hash, source_structure_id: "mock",
    truncated: false, network: { replica: 0, replica_count: 2, gpu_count: 128, local_medium: "NVLink / NVSwitch",
      intra_bandwidth_bytes_s: 450e9, inter_bandwidth_bytes_s: 50e9, iteration: null },
  } }));
  await page.goto(`/workloads/inference/${design.id}?run=${run.id}&view=placement`);
  await expect(page.locator(".wc-gpus button")).toHaveCount(128);
  await expect(page.locator(".wc-placement section")).toHaveCount(16);
  await expect(page.locator(".wc-gpus button").last().locator("meter")).toBeAttached();
  expect(await page.locator(".wc-gpus").evaluateAll(rows => rows.every(row =>
    getComputedStyle(row).gridTemplateColumns.split(" ").length <= 8))).toBe(true);
  expect(await page.locator(".wc-gpus button").evaluateAll(nodes =>
    nodes.every(n => n.scrollWidth <= n.clientWidth))).toBe(true);
  await page.locator(".wc-gpus button").last().scrollIntoViewIfNeeded();
  await page.locator(".wc-canvas-panel").screenshot({ path: `${process.env.PLANNER_E2E_EVIDENCE ?? "/tmp"}/placement-128-mock-${info.project.name}.png` });
  await page.getByRole("button", { name: "Communication", exact: true }).click();
  await expect(page.locator(".wc-network-machine")).toHaveCount(16);
  await expect(page.locator(".wc-network-gpu")).toHaveCount(128);
  await expect(page.locator(".wc-network-replicas")).toHaveCount(1);
  await page.locator(".wc-network-replicas").scrollIntoViewIfNeeded();
  await page.locator(".wc-canvas-panel").screenshot({ path: `${process.env.PLANNER_E2E_EVIDENCE ?? "/tmp"}/network-128-mock-${info.project.name}.png` });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
});
