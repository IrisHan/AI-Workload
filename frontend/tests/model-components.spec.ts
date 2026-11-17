import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const sources = JSON.parse(fs.readFileSync(path.resolve(import.meta.dirname,
  "../../docs/reports/2026-09-13-model-center/evidence/sources/manifest.json"), "utf8")) as
  { repo_id: string; revision: string }[];
const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-13-model-center/evidence/browser";

for (const source of sources) {
  test(`import and inspect ${source.repo_id}`, async ({ page }, info) => {
    test.setTimeout(60000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
    await page.goto("/models/import");
    await page.getByLabel("Repository ID or URL", { exact: true }).fill(`https://huggingface.co/${source.repo_id}`);
    await page.getByLabel("Revision", { exact: true }).fill(source.revision);
    await page.getByRole("button", { name: "Import model", exact: true }).click();
    await expect(page.getByRole("heading", { name: source.repo_id.split("/")[1], exact: true })).toBeVisible({ timeout: 40000 });
    await page.getByRole("tab", { name: "Structure", exact: true }).click();
    await expect(page.getByText("Shadow comparison matched")).toBeVisible();
    await expect(page.locator(".react-flow__node").first()).toBeVisible({ timeout: 15000 });
    if (/V4/.test(source.repo_id)) {
      await page.screenshot({ path: `${evidence}/v4-graph-${info.project.name}.png`, fullPage: false });
    }
    await page.getByRole("button", { name: "参数账本", exact: true }).click();
    await expect(page.getByRole("heading", { name: "模型参数账本" })).toBeVisible();
    if (/V4|vl-7b|Qwen3.8|V3.2|moe-16b/.test(source.repo_id)) {
      await page.screenshot({ path: `${evidence}/${source.repo_id.replace("/", "--")}-${info.project.name}-ledger.png`, fullPage: true });
    }
    await page.getByRole("tab", { name: "Memory", exact: true }).click();
    await expect(page.getByText("Static subtotal", { exact: true })).toBeVisible();
    await expect(page.getByText("Not modeled", { exact: true })).toHaveCount(2);
    await page.getByLabel("Batch size", { exact: true }).fill("2");
    await page.getByRole("button", { name: "Calculate", exact: true }).click();
    await expect(page.getByRole("button", { name: "Calculate", exact: true })).toBeEnabled();
    if (/V4/.test(source.repo_id)) {
      await page.screenshot({ path: `${evidence}/v4-memory-${info.project.name}.png`, fullPage: true });
    }
    const size = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
    expect(size.scroll).toBeLessThanOrEqual(size.width + 1);
    expect(errors).toEqual([]);
  });
}
