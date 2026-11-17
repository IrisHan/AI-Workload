import { test, expect, type APIRequestContext } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { initialInput } from "../src/features/workload-designer/state";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../data/workload-e2e";
async function source(request: APIRequestContext, repo: string, group: string) {
  const metadata = JSON.parse(fs.readFileSync(path.resolve(import.meta.dirname,
    `../../docs/reports/${group}/evidence/sources/${repo.replace("/", "--")}/metadata.json`), "utf8"));
  const response = await request.post("/api/models/import", { data: { repo_id: repo, revision: metadata.sha } });
  expect(response.ok()).toBeTruthy();
  return response.json();
}

test("MiniMax source structure remains interactive while quantized planning is blocked", async ({ page, request }, info) => {
  test.setTimeout(60000);
  const model = await source(request, "MiniMaxAI/MiniMax-M2.5", "2026-09-15-frontier-families");
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto(`/workloads/inference/new?model=${model.id}&hardware=nvidia-h100-sxm-80gb-20260913`);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1440, height: 768 });
  await expect(page.locator(".wc-issues")).toContainText("QUANTIZED_SOURCE");
  await expect(page.getByRole("button", { name: "Auto Design", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Structure", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await page.getByRole("button", { name: "放大", exact: true }).click();
  await page.screenshot({ path: `${evidence}/minimax-blocked-${info.project.name}.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(errors).toEqual([]);
});

test("single GPU OOM exposes rejection without a candidate", async ({ page, request }, info) => {
  const models = await (await request.get("/api/models")).json();
  const model = models.find((m: { repo_id: string }) => m.repo_id === "Qwen/Qwen2.5-0.5B");
  const input = initialInput(model.id, "nvidia-h100-sxm-80gb-20260913");
  input.mode = "offline";
  Object.assign(input.load, { prompt_tokens: 32000, output_tokens: 3, concurrency: 4096, request_count: 4096, request_rate: null, arrival: null });
  input.runtime.max_batch_sequences = 4096;
  input.resources.gpus_per_node = 1; input.resources.topology_template = "single_gpu-v1";
  input.parallelism.mode = "manual";
  input.parallelism.manual_candidate = { tp: 1, pp: 1, ep: 1, dp: 1, boundaries: [] };
  const created = await request.post("/api/workloads/designs", {
    data: { name: `OOM ${info.project.name} ${Date.now()}`, input },
    headers: { "Idempotency-Key": crypto.randomUUID() },
  });
  expect(created.status()).toBe(201);
  await page.goto(`/workloads/inference/${(await created.json()).id}`);
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await expect(page.locator(".wc-run-status")).toContainText("infeasible", { timeout: 15000 });
  await expect(page.locator(".wc-candidates tbody tr")).toHaveCount(0);
  await page.getByText("Rejected Candidates", { exact: true }).click();
  await expect(page.locator(".wc-issues").filter({ hasText: "Rejected Candidates" })).toContainText("CAPACITY");
  await page.screenshot({ path: `${evidence}/oom-${info.project.name}.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
});

test("MoE expert shards reflect EP and DP rank ownership", async ({ page, request }, info) => {
  test.setTimeout(90000);
  const model = await source(request, "deepseek-ai/deepseek-moe-16b-base", "2026-09-13-model-center");
  const input = initialInput(model.id, "nvidia-h100-sxm-80gb-20260913");
  input.mode = "offline";
  Object.assign(input.load, { prompt_tokens: 5, output_tokens: 3, concurrency: 4, request_count: 4, request_rate: null, arrival: null });
  input.runtime.max_batch_sequences = 4;
  input.parallelism.layout = "orthogonal-tp-ep-v1";
  input.parallelism.mode = "manual";
  input.parallelism.manual_candidate = { tp: 2, pp: 1, ep: 2, dp: 2, boundaries: [] };
  const created = await request.post("/api/workloads/designs", {
    data: { name: `MoE shards ${info.project.name} ${Date.now()}`, input },
    headers: { "Idempotency-Key": crypto.randomUUID() },
  });
  expect(created.status()).toBe(201);
  const design = await created.json();
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto(`/workloads/inference/${design.id}`);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1366, height: 768 });
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Structure", exact: true }).click();
  await page.getByLabel("Layer", { exact: true }).selectOption("1");
  await page.getByText("Operator List", { exact: true }).click();
  await page.locator(".wc-node-list button").filter({ hasText: /^routed_expert\.gate_proj / }).click({ timeout: 10000 });
  await expect(page.locator(".wc-inspector")).toContainText("704 × 2048");
  await expect(page.locator(".wc-inspector")).toContainText("Experts [0, 32)");
  await expect(page.locator(".wc-inspector")).toContainText("TP 2 / PP 1 / EP 2 / DP 2");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByLabel("GPU", { exact: true }).selectOption("2");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.locator(".wc-inspector")).toContainText("Experts [32, 64)");
  await page.screenshot({ path: `${evidence}/moe-shards-${info.project.name}.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(errors).toEqual([]);
});
