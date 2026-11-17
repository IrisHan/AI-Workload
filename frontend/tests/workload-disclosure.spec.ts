import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const frozen = JSON.parse(fs.readFileSync(path.join(root,
  "docs/reports/2026-10-08-calculation-explainability/evidence/e2e-release/excel-desktop-export.json"), "utf8"));
const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../data/disclosure-e2e";

test("single GPU progressive disclosure, optimistic assumptions and invariant failure", async ({ page, request }, info) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  const modelResponse = await request.post("/api/models/import", { data: {
    repo_id: frozen.frozen_input.sources.model.repo_id, revision: frozen.frozen_input.sources.model.revision,
  } });
  expect(modelResponse.ok()).toBe(true);
  const model = await modelResponse.json();
  const input = structuredClone(frozen.frozen_input.design);
  input.model_id = model.id;
  const savedResponse = await request.post("/api/workloads/designs", {
    headers: { "Idempotency-Key": crypto.randomUUID() },
    data: { name: `Disclosure ${info.project.name} ${Date.now()}`, input },
  });
  expect(savedResponse.ok()).toBe(true);
  const saved = await savedResponse.json();
  const runResponse = await request.post("/api/workloads/runs", {
    headers: { "Idempotency-Key": crypto.randomUUID() }, data: { design_revision_id: saved.revision.id,
      search: { max_candidates: 1, max_seconds: 20, event_budget: 1000000, top_k: 1 } },
  });
  expect(runResponse.status()).toBe(202);
  const run = await runResponse.json();
  await page.goto(`/workloads/inference/${saved.id}?run=${run.id}&rank=0`);
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  const results = page.getByLabel("Live Results", { exact: true }), dialog = page.getByRole("dialog");
  await expect(results.getByLabel("Decision Summary")).toContainText("1 GPU");
  await expect(results.getByLabel("Why this plan", { exact: true })).toBeVisible();
  await expect(results.locator(".wc-inconsistency")).toHaveCount(0);
  await expect(results.locator(".wc-plan-reasons button")).toHaveCount(11);
  await expect(results.getByLabel("Why this plan", { exact: true }).getByText("Weights", { exact: true })).toHaveCount(1);
  await expect(results.getByLabel("Why this plan", { exact: true })).toContainText("Activations");
  await expect(results.getByLabel("Why this plan", { exact: true })).toContainText("Memory Utilization");
  await expect(results.locator(".wc-expert-raw")).toHaveCount(0);
  fs.mkdirSync(evidence, { recursive: true });
  await page.locator(".wc-workbench").screenshot({ path: `${evidence}/disclosure-summary-${info.project.name}.png` });
  await results.getByRole("button", { name: "Why KV Cache", exact: true }).click();
  await expect(dialog.locator(".wc-template")).toHaveCount(1);
  await expect(dialog).toContainText("28 identical layers");
  await expect(dialog.locator(".wc-formula")).toHaveCount(0);
  await dialog.getByText("Show representative layer", { exact: true }).click();
  await expect(dialog).toContainText("KV Heads");
  await dialog.getByRole("button", { name: "Show Formula", exact: true }).click();
  await expect(dialog.getByLabel("Formula", { exact: true })).toBeVisible();
  await page.screenshot({ path: `${evidence}/disclosure-kv-${info.project.name}.png`, fullPage: true });
  await page.keyboard.press("Escape");
  await results.getByRole("button", { name: "Why Weights", exact: true }).click();
  await expect(dialog.locator(".wc-template")).toHaveCount(1);
  await dialog.getByText("Show weight calculation", { exact: true }).click();
  await expect(dialog).toContainText("q_proj");
  await expect(dialog).toContainText("12,848,640");
  await expect(dialog).not.toContainText("config_sha256");
  await expect(dialog).not.toContainText("Expert Mode");
  await expect(dialog).not.toContainText("Derived · Analytical model");
  await expect(dialog.locator(".wc-evidence-headline")).toContainText("Weights");
  await dialog.getByText("Raw Calculation", { exact: true }).click();
  await expect(dialog).toContainText("config_sha256");
  await page.keyboard.press("Escape");
  await results.getByRole("button", { name: "Why ITL", exact: true }).click();
  await expect(dialog.locator(".wc-facts").first().locator(":scope > div")).toHaveCount(3);
  await expect(dialog).toContainText("this request’s wave service");
  await page.keyboard.press("Escape");
  await results.locator(".wc-performance-bar.is-bottleneck").getByRole("button").click();
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await results.getByText("Advanced · Cost, Energy & Speculative", { exact: true }).click();
  await expect(results.getByLabel("Speculative Cycle", { exact: true })).toContainText("Optimistic assumption");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  // Exercise missing samples and then corruption only in the browser's copy.
  let corruptPeak = false;
  await page.route(`**/api/workloads/runs/${run.id}/plans?*`, async route => {
    const response = await route.fetch(), data = await response.json();
    const summary = data.items[0].summary;
    if (corruptPeak) summary.memory.peak_bytes += 1024;
    else {
      summary.metric_values.iteration_time_ms.value = null;
      summary.performance.metrics.iteration_time_ms = null;
      summary.explanation.traces.iteration_time_ms.result.value = null;
      summary.explanation.performance.bottleneck = "Unavailable";
      for (const component of summary.explanation.performance.breakdown) component.result.value = null;
    }
    await route.fulfill({ response, json: data });
  });
  await page.reload();
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(results.locator(".wc-plan-reasons .wc-result-value").first()).toContainText("Unavailable");
  await expect(results.locator(".wc-plan-reasons .wc-result-value").first()).not.toContainText("0 ms");
  await expect(results.locator(".wc-performance-bar.is-bottleneck")).toHaveCount(0);
  corruptPeak = true;
  await page.reload();
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(results.locator(".wc-inconsistency")).toContainText("Calculation Inconsistency");
  await expect(results.getByLabel("Decision Summary")).toContainText("Feasibility · Calculation Inconsistency");
  expect(errors).toEqual([]);
});
