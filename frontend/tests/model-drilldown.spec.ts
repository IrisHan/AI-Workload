import { test, expect } from "@playwright/test";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-13-model-center/evidence/drilldown";

test("V4 compressed attention and MTP remain navigable", async ({ page, request }, info) => {
  const models = await (await request.get("/api/models")).json();
  const model = models.find((m: { name: string }) => m.name === "DeepSeek-V4-Flash-Base");
  expect(model).toBeDefined();
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  const snapshot = await (await request.post(`/api/models/${model.id}/structure`)).json();
  const group = snapshot.analysis.groups.find((g: { template_id: string }) =>
    snapshot.analysis.templates.find((t: { id: string; attn_kind: string }) =>
      t.id === g.template_id && t.attn_kind === "CSA"));
  const template = snapshot.analysis.templates.find((t: { id: string }) => t.id === group.template_id);
  expect(template.edges).toContainEqual({ source: "q_a_proj", target: "q_norm.weight", label: null });
  expect(template.edges).not.toContainEqual({ source: "q_a_proj", target: "kv_norm.weight", label: null });
  await page.goto(`/models/${model.id}?tab=structure&group=${group.id}&module=indexer.q_proj`);
  await expect(page.getByTestId("module-detail")).toContainText("8,388,608");
  if (info.project.name === "mobile")
    await page.getByRole("button", { name: "画布", exact: true }).click();
  await page.getByRole("button", { name: "Attention", exact: true }).click();
  await expect(page.locator('.react-flow__node[data-id="indexer.score"]')).toBeVisible();
  await page.locator(".react-flow").screenshot({ path: `${evidence}/v4-canvas-${info.project.name}.png` });
  await page.getByRole("checkbox", { name: "MTP", exact: true }).check();
  await page.getByRole("button", { name: "应用配置", exact: true }).click();
  await expect(page.getByText("Parameter delta: 6610048891", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "参数账本", exact: true }).click();
  await page.getByRole("button", { name: "mtp0", exact: true }).click();
  await expect(page.locator(".canvas-heading h2")).toHaveText("MTP");
  await expect(page.locator('.react-flow__node[data-id="e_proj"]')).toBeVisible();
  await page.locator(".react-flow").screenshot({ path: `${evidence}/v4-mtp-${info.project.name}.png` });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  expect(errors).toEqual([]);
});
