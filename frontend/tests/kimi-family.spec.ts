import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-15-kimi-family/evidence";
const sourcesRoot = path.resolve(import.meta.dirname, "../../docs/reports/2026-09-15-kimi-family/evidence/sources");
test.setTimeout(90000);

const TARGETS = [
  { name: "Kimi-K2-Instruct", folder: "moonshotai--Kimi-K2-Instruct" },
  { name: "Kimi-VL-A3B-Instruct", folder: "moonshotai--Kimi-VL-A3B-Instruct" },
  { name: "Kimi-K2.5", folder: "moonshotai--Kimi-K2.5" },
  { name: "Kimi-K3", folder: "moonshotai--Kimi-K3" },
];

function pinned(folder: string) {
  const meta = JSON.parse(fs.readFileSync(path.join(sourcesRoot, folder, "metadata.json"), "utf8")) as {
    id: string;
    sha: string;
  };
  return meta;
}

async function ensureImported(
  page: import("@playwright/test").Page,
  request: import("@playwright/test").APIRequestContext,
  folder: string,
  name: string,
) {
  const models = (await (await request.get("/api/models")).json()) as { id: string; name: string }[];
  const existing = models.find((item) => item.name === name);
  if (existing) return existing;
  const meta = pinned(folder);
  await page.goto("/models/import");
  await page.getByLabel("Repository ID or URL", { exact: true }).fill(`https://huggingface.co/${meta.id}`);
  await page.getByLabel("Revision", { exact: true }).fill(meta.sha);
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await expect(page.getByRole("heading", { name, exact: true })).toBeVisible({ timeout: 60000 });
  const after = (await (await request.get("/api/models")).json()) as { id: string; name: string }[];
  const model = after.find((item) => item.name === name);
  expect(model, `imported ${name}`).toBeTruthy();
  return model!;
}

for (const target of TARGETS) {
  test(`import and inspect ${target.name}`, async ({ page, request }, info) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    const model = await ensureImported(page, request, target.folder, target.name);
    await page.goto(`/models/${model.id}`);
    await expect(page.getByRole("heading", { name: target.name, exact: true })).toBeVisible();
    await expect(page.getByRole("link", { name: "Create workload" })).toBeVisible();
    await expect(page.getByText("morphology-workload-1.1.0").first()).toBeVisible();
    await page.getByRole("tab", { name: "Structure", exact: true }).click();
    await expect(page.locator(".react-flow__node").first()).toBeVisible({ timeout: 20000 });
    await expect(page.getByTestId("morphology-tags")).toBeVisible();
    await expect(page.getByTestId("ffn-recipe")).toContainText("MoE");
    await expect(page.locator(".react-flow__node").filter({ hasText: "router" }).first()).toBeVisible();
    if (target.name === "Kimi-K3")
      await expect(page.getByTestId("ffn-recipe")).toContainText("latent");
    if (info.project.name === "desktop" || info.project.name === "mobile") {
      await page.screenshot({
        path: `${evidence}/${target.name.toLowerCase()}-structure-${info.project.name}.png`,
        fullPage: false,
      });
    }
    await page.getByRole("tab", { name: "Memory", exact: true }).click();
    await expect(page.getByText("Static subtotal", { exact: true })).toBeVisible();
    const size = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
    expect(size.scroll).toBeLessThanOrEqual(size.width + 1);
    expect(errors).toEqual([]);
  });
}
