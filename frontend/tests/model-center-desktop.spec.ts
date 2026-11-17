import { test, expect } from "@playwright/test";

const evidence =
  process.env.PLANNER_E2E_EVIDENCE ??
  "../docs/reports/2026-09-15-model-center-desktop/evidence";
const qwen = "2cb255f7a2a546a14e2923f7";

test("desktop nav collapse, memory compact row and structure focus page", async ({
  page,
}, info) => {
  test.skip(info.project.name !== "desktop", "Desktop chrome density is the acceptance target.");
  test.setTimeout(60000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/models");
  const workspace = page.locator(".app-shell > .workspace");
  const expanded = (await workspace.boundingBox())!.width;
  await page.getByRole("button", { name: "收起导航" }).click();
  await expect(page.locator(".app-shell")).toHaveClass(/nav-collapsed/);
  const collapsed = (await workspace.boundingBox())!.width;
  expect(collapsed - expanded).toBeGreaterThan(150);
  await page.screenshot({ path: `${evidence}/nav-collapsed-${info.project.name}.png` });
  await page.getByRole("button", { name: "展开导航" }).click();
  await expect(page.getByRole("button", { name: "收起导航" })).toBeVisible();

  await page.goto(`/models/${qwen}?tab=memory`);
  await expect(page.getByRole("heading", { name: "Static memory" })).toBeVisible();
  const batch = await page.getByRole("spinbutton", { name: "Batch size" }).boundingBox();
  const context = await page.getByRole("spinbutton", { name: "Context tokens" }).boundingBox();
  const dtype = await page.getByRole("combobox", { name: "Resident dtype" }).boundingBox();
  expect(batch).toBeTruthy();
  expect(context).toBeTruthy();
  expect(dtype).toBeTruthy();
  expect(Math.abs(batch!.y - context!.y)).toBeLessThan(8);
  expect(Math.abs(context!.y - dtype!.y)).toBeLessThan(8);
  expect(batch!.width).toBeLessThan(200);
  expect(context!.width).toBeLessThan(200);
  expect(dtype!.width).toBeLessThan(200);
  await page.screenshot({ path: `${evidence}/memory-row-${info.project.name}.png` });

  await page.goto(`/models/${qwen}?tab=structure`);
  const studio = page.getByRole("region", { name: "Spec Studio model workspace" });
  await expect(studio.locator(".react-flow__node").first()).toBeVisible();
  const embedded = await studio.locator(".flow-wrap").boundingBox();
  const pop = studio.getByRole("link", { name: "在独立页面打开结构图" });
  await expect(pop).toHaveAttribute("href", /\/models\/.+\/studio/);
  await page.getByRole("button", { name: "收起导航" }).click();
  const afterNav = await studio.locator(".flow-wrap").boundingBox();
  expect(afterNav!.width).toBeGreaterThan(embedded!.width);

  const popupPromise = page.waitForEvent("popup");
  await pop.click();
  const popup = await popupPromise;
  await popup.waitForLoadState("domcontentloaded");
  await expect(popup.locator(".app-shell")).toHaveClass(/chrome-focus/);
  await expect(popup.getByRole("link", { name: "返回模型详情" })).toBeVisible();
  await expect(popup.locator(".react-flow__node").first()).toBeVisible();
  await popup.screenshot({ path: `${evidence}/studio-popup-${info.project.name}.png` });
  await popup.close();

  await page.goto(`/models/${qwen}/studio?tab=structure`);
  await expect(page.locator(".app-shell")).toHaveClass(/chrome-focus/);
  await expect(page.getByRole("link", { name: "在独立页面打开结构图" })).toHaveCount(0);
  const focused = await page.locator(".spec-studio .flow-wrap").boundingBox();
  expect(focused!.height).toBeGreaterThan(embedded!.height);
  expect(focused!.width).toBeGreaterThan(embedded!.width);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    page.viewportSize()!.width + 1,
  );
  await page.screenshot({ path: `${evidence}/studio-focus-${info.project.name}.png` });
  expect(errors).toEqual([]);
});
