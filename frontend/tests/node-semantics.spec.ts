import { test, expect } from "@playwright/test";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-13-structure-node-semantics/evidence";
const model = "2cb255f7a2a546a14e2923f7";
const historical = "e054cd953dc1d1c47e9c379c309c477e7199c7606e22e51851e2140863ca68e4";

test("head views distinguish weights from activations and preserve model heading", async ({ page, request }, info) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  const oldResponse = await request.get(`/api/models/${model}/structures/${historical}`);
  const oldSnapshot = oldResponse.ok() ? await oldResponse.json() : null;
  const created = await (await request.post(`/api/models/${model}/structure`)).json();
  const repeated = await (await request.post(`/api/models/${model}/structure`)).json();
  expect(repeated).toEqual(created);
  if (oldSnapshot) {
    expect(await (await request.get(`/api/models/${model}/structures/${historical}`)).json()).toEqual(oldSnapshot);
  }
  for (const [node, projection, shape, params] of [
    ["q_heads", "q_proj", "[B, S, 12, 128]", "2,360,832"],
    ["k_heads", "k_proj", "[B, S, 2, 128]", "393,472"],
  ]) {
    await page.goto(`/models/${model}?tab=structure&snapshot=${created.id}&group=g1&module=${node}`);
    await expect(page.locator("h1")).toHaveCount(1);
    await expect(page.locator("h1")).toHaveText("Qwen2.5-1.5B");
    await expect(page.getByRole("button", { name: "导出分析 JSON" })).toBeVisible();
    const detail = page.getByTestId("operator-detail");
    await expect(detail).toContainText("0 · 无独立权重");
    await expect(detail).toContainText(shape);
    await expect(detail).toContainText("可能共享输入存储");
    await expect(detail).toContainText("未评估，不能由参数量推断");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
    if (info.project.name === "desktop")
      await expect(page.locator(".react-flow__node").first()).toBeVisible();
    await page.screenshot({ path: `${evidence}/${node}-${info.project.name}.png`, fullPage: true });
    await detail.getByRole("button", { name: projection, exact: true }).click();
    await expect(page.getByTestId("module-detail")).toContainText(params);
    await expect(page.getByTestId("module-detail")).toContainText("非激活张量");
  }
  const studio = page.getByRole("region", { name: "Spec Studio model workspace" });
  const close = studio.getByRole("button", { name: "关闭详情" });
  if (await close.isVisible()) await close.click();
  await studio.getByRole("button", { name: "Attention", exact: true }).click();
  await expect(studio.locator(".module-table tbody tr")).toHaveCount(4);
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".module-table tbody tr")).toHaveCount(3);
  await expect(studio.locator(".module-table")).not.toContainText("q_proj");
  await studio.getByRole("button", { name: "完整层", exact: true }).click();
  await expect(studio.locator(".module-table tbody tr")).toHaveCount(9);
  await expect(studio.locator('.react-flow__node[data-id="q_heads"]')).toContainText("张量视图 · 无独立权重");
  await studio.getByRole("button", { name: "本组 ×28", exact: true }).click();
  await expect(studio.locator(".module-table tr").filter({ hasText: "q_proj" })).toContainText("66,103,296");
  await page.goto(`/models/${model}?tab=structure&snapshot=${created.id}&shared=lm_head&module=lm_head`);
  if (await studio.getByRole("button", { name: "详情", exact: true }).isVisible())
    await studio.getByRole("button", { name: "详情", exact: true }).click();
  await expect(page.getByTestId("module-detail")).toContainText("不代表没有使用权重");
  expect(errors).toEqual([]);
});

test("unmodeled graph node must not claim zero parameters", async ({ page, request }) => {
  const snapshot = await (await request.post(`/api/models/${model}/structure`)).json();
  // Synthetic display-only fixture; it is not persisted as a model or analysis.
  snapshot.analysis.templates[0].nodes.push({
    id: "unknown_test_node", label: "Unknown synthetic node", category: "other",
  });
  await page.route(`**/api/models/${model}/structure`, route => route.fulfill({ json: snapshot }));
  await page.goto(`/models/${model}?tab=structure&module=unknown_test_node`);
  const detail = page.getByTestId("operator-detail");
  await expect(detail).toContainText("未提供独立参数账本");
  await expect(detail).not.toContainText("0 · 无独立权重");
});
