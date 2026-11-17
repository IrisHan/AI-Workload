import { test, expect } from "@playwright/test";
import fs from "node:fs";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../data/workload-e2e";
test("Excel A100 spec mock: sections, editable fields, framework PP, cycle and persistence", async ({ page, request }, info) => {
  test.setTimeout(150000);
  page.setDefaultTimeout(12000);
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  const imported = await request.post("/api/models/import", { data: {
    repo_id: "Qwen/Qwen2.5-7B", revision: "d149729398750b98c0af14eb82c78cfe92750796",
  } });
  expect(imported.ok()).toBeTruthy();
  const model = await imported.json();
  const name = `mock Excel A100+spec ${info.project.name} ${Date.now()}`;
  await page.goto(`/workloads/inference/new?model=${model.id}&hardware=nvidia-a100-sxm-80gb-20260913`);
  await page.setViewportSize(info.project.name === "desktop" ? { width: 1366, height: 768 } : { width: 390, height: 844 });
  await expect(page.locator(".wc-zone-title h2")).toHaveText(["Model & Hardware", "Workload Configuration", "Analysis Workspace"]);
  await page.getByLabel("Design Name").fill(name);
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  const sections = page.locator(".wc-input > details");
  await expect(sections).toHaveCount(5);
  for (let i = 0; i < 5; i++) {
    const section = sections.nth(i), summary = section.locator(":scope > summary");
    await summary.click();
    await expect(section).not.toHaveAttribute("open");
    await expect(summary).toBeVisible();
  }
  await page.screenshot({ path: `${evidence}/framework-collapsed-${info.project.name}.png`, fullPage: true });
  for (let i = 4; i >= 0; i--) {
    await sections.nth(i).locator(":scope > summary").click();
    await expect(sections.nth(i)).toHaveAttribute("open");
  }
  await expect(page.getByLabel("Draft Tokens", { exact: true })).toHaveCount(0);
  await page.getByLabel("Mode", { exact: true }).selectOption("offline");
  for (const [label, value] of Object.entries({
    "Prompt Tokens": "2048", "Output Tokens": "281", "Batch Size": "16",
    "Concurrency": "16", "Request Count": "16", "Prefill Chunk": "256",
    "Intra Injection (GB/s)": "50", "Inter / GPU (GB/s)": "50",
  })) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByLabel("Search Mode", { exact: true }).selectOption("manual");
  await page.getByLabel("TP", { exact: true }).fill("2");
  await expect(page.getByText("Pipeline layers: Stage 0: 0–27", { exact: true })).toBeVisible();
  await expect(page.getByLabel("PP Boundaries", { exact: true })).toHaveCount(0);
  await page.getByLabel("PP", { exact: true }).fill("3");
  await expect(page.getByLabel("PP Boundaries", { exact: true })).toHaveValue("9,19");
  await page.getByLabel("Framework", { exact: true }).selectOption("sglang-v0.5.3");
  await page.getByLabel("PP", { exact: true }).fill("3");
  await expect(page.getByLabel("PP Boundaries", { exact: true })).toHaveValue("9,18");
  await page.getByLabel("Framework", { exact: true }).selectOption("vllm-v0.10.2");
  await page.getByLabel("GPUs / Node", { exact: true }).fill("1");
  await page.getByLabel("Speculative Decoding", { exact: true }).check();
  for (const [label, value] of Object.entries({
    "Draft Tokens": "2", "Acceptance Rate": "0.7", "Draft Latency (ms)": "0", "Draft Memory (GB)": "0",
  })) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByLabel("Include one Prefill Chunk in cycle", { exact: true }).check();
  await expect(page.locator(".wc-readout").filter({ hasText: "Expected Accepted Tokens" }).locator("output")).toHaveText("1.19");
  await expect(page.locator(".wc-readout").filter({ hasText: "Expected Emitted Tokens" }).locator("output")).toHaveText("2.19");
  await expect(page.locator(".wc-readout input, .wc-hardware-facts input")).toHaveCount(0);
  await page.getByLabel("Activation Efficiency", { exact: true }).fill("");
  await page.getByLabel("Activation Efficiency", { exact: true }).pressSequentially("0.85");
  await page.getByLabel("Activation Efficiency", { exact: true }).blur();
  await expect(page.getByLabel("Activation Efficiency", { exact: true })).toHaveValue("0.85");
  await page.getByRole("button", { name: "Override · Reset", exact: true }).click();
  await expect(page.getByLabel("Activation Efficiency", { exact: true })).toHaveValue("1");
  await page.getByLabel("Compute Efficiency", { exact: true }).fill("0.7");
  await page.getByLabel("HBM Efficiency", { exact: true }).fill("0.7");
  await page.getByText("Runtime Assumptions", { exact: true }).click();
  await page.getByLabel("hbm_bandwidth_gb_s", { exact: true }).fill("2000");
  await page.getByText("Runtime Assumptions", { exact: true }).click();
  await page.getByText("Runtime Policy & Measurement Window", { exact: true }).click();
  const fieldName = page.locator(".wc-field-name").filter({ hasText: /^Duration \(s\)/ });
  await fieldName.focus();
  await expect(fieldName.getByRole("tooltip")).toBeVisible();
  await expect(fieldName.getByRole("tooltip")).toContainText("观察窗口");
  await page.getByLabel("Duration (s)", { exact: true }).focus();
  await page.getByText("Runtime Policy & Measurement Window", { exact: true }).click();
  await page.screenshot({ path: `${evidence}/framework-inputs-${info.project.name}.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 90000 });
  await expect(page.locator(".wc-run-status")).toContainText("feasible");
  const url = page.url(), runId = new URL(url).searchParams.get("run")!;
  const designId = new URL(url).pathname.split("/").at(-1)!;
  const exported = await (await request.get(`/api/workloads/runs/${runId}/export`)).json();
  const cycle = exported.plans[0].summary.speculative_cycle;
  expect(cycle.expected_emitted_tokens).toBeCloseTo(2.19, 12);
  expect(cycle.decode_requests).toBe(16);
  expect(cycle.prefill_tokens).toBe(256);
  expect(cycle.capacity_status).toBe("pass");
  expect(cycle.cycle_seconds).toBeGreaterThan(0);
  fs.mkdirSync(evidence, { recursive: true });
  fs.writeFileSync(`${evidence}/excel-${info.project.name}-export.json`, JSON.stringify(exported, null, 2));
  fs.writeFileSync(`${evidence}/excel-${info.project.name}-browser.json`, JSON.stringify({ url, designId, runId, name, errors }, null, 2));
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await page.getByText("Advanced · Cost, Energy & Speculative", { exact: true }).click();
  await expect(page.getByLabel("Speculative Cycle", { exact: true })).toContainText("2.19");
  await expect(page.getByLabel("Speculative Cycle", { exact: true })).toContainText("Optimistic assumption");
  await page.screenshot({ path: `${evidence}/framework-results-${info.project.name}.png`, fullPage: true });
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Export JSON", exact: true }).click();
  await (await download).saveAs(`${evidence}/excel-${info.project.name}-download.json`);
  await page.reload();
  await expect(page.locator(".wc-run-status")).toContainText("completed");
  await expect(page.getByLabel("Design Name")).toHaveCount(0);
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  await expect(page.getByLabel("Acceptance Rate", { exact: true })).toHaveValue("0.7");
  await page.getByLabel("Speculative Decoding", { exact: true }).uncheck();
  await expect(page.getByLabel("Draft Tokens", { exact: true })).toHaveCount(0);
  await page.getByLabel("Speculative Decoding", { exact: true }).check();
  await expect(page.getByLabel("Draft Tokens", { exact: true })).toHaveValue("2");
  await page.getByRole("button", { name: "Save as New Design", exact: true }).click();
  await expect(page).not.toHaveURL(new RegExp(designId));
  const fork = await (await request.get(`/api/workloads/designs/${new URL(page.url()).pathname.split("/").at(-1)}`)).json();
  expect(fork.revision.input).toEqual(exported.frozen_input.design);
  expect(fork.runs).toEqual([]);
  expect(errors).toEqual([]);
});
