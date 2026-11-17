import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../data/p1-e2e";
test("focus cost wheel and End reach final row and assumptions", async ({ page, request }, info) => {
  test.setTimeout(60000);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1440, height: 900 });
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  let models = await (await request.get("/api/models")).json() as { id: string; repo_id: string }[];
  let model = models.find(m => m.repo_id === "openai/gpt-oss-120b");
  if (!model) {
    const source = path.resolve(import.meta.dirname,
      "../../docs/reports/2026-09-15-frontier-families/evidence/sources/openai--gpt-oss-120b/metadata.json");
    const meta = JSON.parse(fs.readFileSync(source, "utf8")) as { sha: string };
    const imported = await request.post("/api/models/import", {
      data: { repo_id: "openai/gpt-oss-120b", revision: meta.sha },
    });
    expect(imported.ok()).toBeTruthy();
    models = await (await request.get("/api/models")).json();
    model = models.find(m => m.repo_id === "openai/gpt-oss-120b");
  }
  expect(model).toBeTruthy();
  await page.goto(`/models/${model!.id}/studio?studio_view=cost`);
  await page.getByRole("button", { name: "计算", exact: true }).click();
  await expect(page.locator(".cost-summary")).toContainText("operator-cost-2.2.0");
  const content = page.getByRole("main", { name: "算子成本内容" });
  const lastRow = content.locator("tbody tr").last();
  const lastAssumption = content.locator(".byte-ledger li").last();
  await content.hover();
  await page.mouse.wheel(0, 8000);
  await expect.poll(() => content.evaluate(el => el.scrollTop)).toBeGreaterThan(0);
  await expect(lastAssumption).toBeInViewport();
  // Keyboard alone must also reach the tail after returning to the top.
  await content.focus();
  await page.keyboard.press("Home");
  await expect.poll(() => content.evaluate(el => el.scrollTop)).toBe(0);
  await page.keyboard.press("End");
  await expect(lastAssumption).toBeInViewport();
  const geometry = await content.evaluate(el => ({
    scrollTop: el.scrollTop, height: el.clientHeight, contentHeight: el.scrollHeight,
    viewport: { width: innerWidth, height: innerHeight },
    pageWidth: document.documentElement.scrollWidth,
  }));
  expect(geometry.pageWidth).toBeLessThanOrEqual(geometry.viewport.width + 1);
  await lastRow.scrollIntoViewIfNeeded();
  await expect(lastRow).toBeInViewport();
  fs.mkdirSync(evidence, { recursive: true });
  fs.writeFileSync(`${evidence}/p1-scroll-${info.project.name}.json`, JSON.stringify({ geometry, errors }, null, 2));
  await page.screenshot({ path: `${evidence}/p1-cost-tail-${info.project.name}.png` });
  expect(errors).toEqual([]);
});
