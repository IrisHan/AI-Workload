import { test, expect } from "@playwright/test";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-10-06-workload-implementation/evidence/resume";
test("inference structure zoom, sharded inspection, stale input, history and export", async ({ page, request }, info) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  const models = await (await request.get("/api/models")).json();
  const model = models.find((m: { repo_id: string }) => m.repo_id === "Qwen/Qwen2.5-0.5B");
  await page.goto(`/workloads/inference/new?model=${model.id}&hardware=nvidia-h100-sxm-80gb-20260913`);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1366, height: 768 });
  await expect(page.getByRole("button", { name: "Params", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Structure", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await page.getByLabel("Design Name").fill(`Inference browser ${info.project.name} ${Date.now()}`);
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  await page.getByLabel("Mode", { exact: true }).selectOption("offline");
  await page.getByLabel("Prompt Tokens", { exact: true }).fill("5");
  await page.getByLabel("Output Tokens", { exact: true }).fill("3");
  await page.getByLabel("Concurrency", { exact: true }).fill("2");
  await page.getByLabel("Batch Size", { exact: true }).fill("2");
  await page.getByLabel("Request Count", { exact: true }).fill("2");
  await page.getByLabel("GPUs / Node", { exact: true }).fill("2");
  await page.getByLabel("Search Mode", { exact: true }).selectOption("manual");
  await page.getByLabel("TP", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Structure", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  const viewport = page.locator(".react-flow__viewport");
  const before = await viewport.getAttribute("style");
  await page.getByRole("button", { name: "放大", exact: true }).click();
  await expect(viewport).not.toHaveAttribute("style", before!);
  await page.getByText("Operator List", { exact: true }).click();
  await page.locator(".wc-node-list button").filter({ hasText: /^q_proj$/ }).click();
  await expect(page.locator(".wc-inspector")).toContainText("q_proj");
  await expect(page.locator(".wc-inspector")).toContainText("448 × 896");
  await expect(page.locator(".wc-inspector")).toContainText("Layer KV Cache");
  if (info.project.name === "desktop") {
    await page.locator(".react-flow__node-block").filter({ hasText: "q_proj" }).click();
    await expect(page.locator(".wc-inspector h2")).toHaveText("q_proj");
    const point = await page.locator(".react-flow__pane").evaluate(pane => {
      const box = pane.getBoundingClientRect();
      for (const u of [0.8, 0.6, 0.4, 0.2]) for (const v of [0.5, 0.3, 0.7]) {
        const x = box.x + box.width * u, y = box.y + box.height * v;
        if (document.elementFromPoint(x, y) === pane) return { x, y };
      }
      return null;
    });
    expect(point).not.toBeNull();
    const transform = await viewport.getAttribute("style");
    await page.mouse.move(point!.x, point!.y);
    await page.mouse.down();
    await page.mouse.move(point!.x - 50, point!.y + 35, { steps: 8 });
    await page.mouse.up();
    await expect(viewport).not.toHaveAttribute("style", transform!);
  }
  await page.screenshot({ path: `${evidence}/inference-${info.project.name}.png`, fullPage: true });
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByLabel("Phase", { exact: true }).selectOption("decode");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.locator(".wc-inspector")).toContainText("decode");
  await page.getByRole("button", { name: "Memory Timeline", exact: true }).click();
  await expect(page.getByRole("img", { name: "Modeled GPU memory over time" })).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Export JSON", exact: true }).click();
  expect((await download).suggestedFilename()).toMatch(/^workload-.*\.json$/);
  const runUrl = page.url();
  await page.reload();
  await expect(page.locator(".wc-run-status")).toContainText("completed");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  await page.getByLabel("Batch Size", { exact: true }).fill("1");
  await expect(page.locator(".wc-designer > .wc-notice")).toContainText("Stale Result");
  const sizes = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(sizes.scroll).toBeLessThanOrEqual(sizes.viewport + 1);
  expect(errors).toEqual([]);
  expect(runUrl).toContain("/workloads/inference/");
});
