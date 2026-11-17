import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const frozen = JSON.parse(fs.readFileSync(path.join(root,
  "docs/reports/2026-10-08-calculation-explainability/evidence/original-export.json"), "utf8"));
const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../data/explainability-e2e";

test("acceptance TP2 calculation story, trace, operator and unchanged placement", async ({ page, request }, info) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  const modelResponse = await request.post("/api/models/import", { data: {
    repo_id: frozen.frozen_input.sources.model.repo_id, revision: frozen.frozen_input.sources.model.revision,
  } });
  expect(modelResponse.ok(), await modelResponse.text()).toBeTruthy();
  const model = await modelResponse.json();
  const savedResponse = await request.post("/api/workloads/designs", {
    headers: { "Idempotency-Key": crypto.randomUUID() },
    data: { name: `Calculation acceptance ${info.project.name} ${Date.now()}`,
      input: { ...frozen.frozen_input.design, model_id: model.id } },
  });
  expect(savedResponse.ok(), await savedResponse.text()).toBeTruthy();
  const saved = await savedResponse.json();
  const runResponse = await request.post("/api/workloads/runs", {
    headers: { "Idempotency-Key": crypto.randomUUID() },
    data: { design_revision_id: saved.revision.id,
      search: { max_candidates: 1, max_seconds: 20, event_budget: 1000000, top_k: 1 } },
  });
  expect(runResponse.status()).toBe(202);
  const run = await runResponse.json();
  await page.goto(`/workloads/inference/${saved.id}?run=${run.id}&rank=0`);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1366, height: 768 });
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  const exported = await (await request.get(`/api/workloads/runs/${run.id}/export`)).json();
  const plan = exported.plans[0];
  expect(plan.summary.gpu_count).toBe(2);
  expect(plan.summary.performance.metrics.output_tokens_s).toBe(72.8);
  expect(plan.summary.memory.peak_bytes).toBe(17136715399);
  expect(plan.summary.explanation.scope.run_id).toBe(run.id);
  fs.mkdirSync(evidence, { recursive: true });
  fs.writeFileSync(`${evidence}/acceptance-${info.project.name}-export.json`, JSON.stringify(exported));
  fs.writeFileSync(`${evidence}/acceptance-${info.project.name}-url.json`, JSON.stringify({ url: page.url(), design: saved.id, run: run.id }));
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  const results = page.getByLabel("Live Results", { exact: true });
  await expect(results.locator(".wc-execution-summary")).toContainText("Qwen2.5-7B-Instruct-1M");
  await expect(results.locator(".wc-execution-summary")).toContainText("TP 2 / PP 1 / EP 1 / DP 1");
  await expect(results.getByLabel("GPU Memory", { exact: true })).toContainText("74.506 GiB");
  await expect(results.locator(".wc-memory-summary meter")).toHaveAttribute("value", "0.2142089424875");
  await expect(results.locator(".wc-calculation[open]")).toHaveCount(0);
  await expect(results.locator(".wc-assumptions")).not.toHaveAttribute("open");
  await page.screenshot({ path: `${evidence}/summary-${info.project.name}.png`, fullPage: true });
  await page.locator(".wc-workbench").screenshot({ path: `${evidence}/workspace-${info.project.name}.png` });
  await expect(results.getByLabel("Decision Summary")).toBeVisible();
  await expect(results.locator(".wc-expert-raw")).toHaveCount(0);
  await expect(results).not.toContainText(plan.result_hash);
  const kvButton = results.getByRole("button", { name: "Why KV Cache", exact: true });
  await kvButton.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("238.116 MiB");
  await expect(dialog.locator(".wc-template")).toHaveCount(1);
  await expect(dialog).toContainText("28 identical layers");
  await expect(dialog.locator(".wc-formula")).toHaveCount(0);
  await dialog.getByRole("button", { name: "Show Formula", exact: true }).click();
  await expect(dialog.locator(".wc-formula")).toContainText("kv_efficiency");
  await expect(dialog).not.toContainText("Expert Mode");
  await expect(dialog).not.toContainText("Derived · Analytical model");
  await expect(dialog.locator(".wc-evidence-headline")).toBeVisible();
  await dialog.getByText("Raw Calculation", { exact: true }).click();
  await expect(dialog).toContainText("config_sha256");
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(kvButton).toBeFocused();
  await results.getByRole("button", { name: "Why ITL", exact: true }).click();
  await expect(dialog).toContainText("38.005");
  await dialog.getByRole("button", { name: "Close evidence" }).click();
  await results.getByRole("button", { name: "Why Output Throughput", exact: true }).click();
  await expect(dialog).toContainText("2048 completed-request tokens + 136 unfinished-request tokens");
  await dialog.getByRole("button", { name: "Close evidence" }).click();
  await expect(results).toContainText("Current Bottleneck · Memory Bound");
  await page.screenshot({ path: `${evidence}/performance-${info.project.name}.png`, fullPage: true });
  await results.getByText("Assumptions", { exact: true }).click();
  await results.getByText("Hardware Spec & Effective Model", { exact: true }).click();
  await expect(results.locator(".wc-assumptions")).toContainText("25 GB/s/GPU");
  await expect(results.locator(".wc-assumptions")).toContainText("Override");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Structure", exact: true }).click();
  await expect(page.locator(".wc-graph-context")).toContainText("2048 local tokens");
  await expect(page.locator(".wc-toolbar").first().getByRole("button", { name: /^(Params|Structure|Placement|Communication)$/ })).toHaveCount(4);
  await page.getByText("Operator List", { exact: true }).click();
  await page.locator(".wc-node-list button").filter({ hasText: /^gate_proj$/ }).click();
  const inspector = page.getByLabel("Selection details");
  await expect(inspector).toContainText("9472 × 3584");
  await expect(inspector).toContainText("33,947,648");
  await expect(inspector).toContainText("0.896 ms");
  await page.locator(".wc-workbench").screenshot({ path: `${evidence}/operator-overview-${info.project.name}.png` });
  await inspector.getByRole("button", { name: "Why gate_proj Latency", exact: true }).click();
  await expect(dialog).toContainText("139,049,566,208");
  await dialog.getByRole("button", { name: "Close evidence" }).click();
  await inspector.getByLabel("Kernel Step", { exact: true }).press("End");
  await inspector.getByLabel("Kernel Step", { exact: true }).press("ArrowLeft");
  await inspector.getByLabel("Kernel Step", { exact: true }).press("ArrowLeft");
  await expect(inspector).toContainText("139 MiB");
  await page.screenshot({ path: `${evidence}/operator-${info.project.name}.png`, fullPage: true });
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Placement", exact: true }).click();
  await expect(page.locator(".wc-gpus button")).toHaveCount(2);
  await page.screenshot({ path: `${evidence}/placement-${info.project.name}.png`, fullPage: true });
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  await page.getByLabel("Batch Size", { exact: true }).fill("3");
  await expect(page.locator(".wc-designer > .wc-notice")).toContainText("Stale Result");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await page.getByRole("button", { name: "Live Results", exact: true }).click();
  await results.getByText("Calculation Context", { exact: true }).click();
  await expect(results.getByLabel("Calculation Context", { exact: true })).toContainText("Batch 4");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(errors).toEqual([]);
});
