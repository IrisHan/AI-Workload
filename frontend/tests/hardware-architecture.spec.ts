import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-14-hardware-architecture/evidence";
const examples = [
  ["nvidia-a100-pcie-80gb", "6,912", "ampere"],
  ["nvidia-a100-sxm-80gb", "6,912", "ampere"],
  ["nvidia-h100-sxm-80gb", "16,896", "hopper"],
  ["nvidia-h100-pcie-80gb", "14,592", "hopper"],
  ["nvidia-h20-sxm-96gb", "约9,984", "hopper"],
  ["nvidia-l20-pcie-48gb", "11,776", "ada"],
  ["nvidia-l40-pcie-48gb", "18,176", "ada"],
  ["nvidia-b200-sxm-180gb", "18,944", "blackwell"],
  ["nvidia-b300-sxm-270gb", "20,480", "blackwell-ultra"],
];

test("architecture eight panels across GPU variants", async ({ page }, info) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", e => { if (e.type() === "error") errors.push(e.text()); });
  for (const [sku, cores] of examples) {
    await page.goto(`/hardware/${sku}-20260913?tab=architecture`);
    await expect(page.locator(".arch-section")).toHaveCount(8);
    await expect(page.locator('.arch-section-2 [data-fact="cuda_count"] strong')).toHaveText(cores);
    await expect(page.locator('.arch-memory [data-fact="register"] strong')).toHaveText("256 KiB");
    const dimension = await page.evaluate(() => ({
      page: document.documentElement.scrollWidth, viewport: innerWidth,
      white: getComputedStyle(document.body).backgroundColor,
    }));
    expect(dimension.page).toBeLessThanOrEqual(dimension.viewport + 1);
    expect(await page.locator("h1").count()).toBe(1);
    await page.screenshot({ path: `${evidence}/${sku}-${info.project.name}.png`, fullPage: true });
  }
  expect(errors).toEqual([]);
});

test("architecture clean display, export preserves audit and SKU navigation", async ({ page }) => {
  await page.goto("/hardware/nvidia-b200-sxm-180gb-20260913?tab=architecture");
  await expect(page.locator(".arch-section")).toHaveCount(8);
  await expect(page.locator('.arch-memory [data-fact="l1_pool"] strong')).toHaveText("256 KiB");
  await expect(page.locator(".arch-notes, .arch-reference-label, .arch-fact-scope")).toHaveCount(0);
  await expect(page.locator(".architecture-atlas details, .architecture-atlas summary")).toHaveCount(0);
  await expect(page.locator(".arch-comparison table")).toBeVisible();
  await expect(page.locator(".arch-chip-table table")).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByTitle("Export architecture JSON").click();
  const file = await download;
  expect(file.suggestedFilename()).toContain("-architecture.json");
  const payload = JSON.parse(await readFile((await file.path())!, "utf8"));
  expect(payload.sections[2].notes.join(" ")).toContain("附件把SM10.x写成128KB");
  expect(payload.sections[2].reference_text).toContain("128 KB/SM");
  await page.goto("/hardware/nvidia-h100-sxm-80gb-20260913?tab=architecture");
  await page.locator(".arch-related").getByRole("link", { name: "H100 PCIe 80GB" }).click();
  await expect(page).toHaveURL(/nvidia-h100-pcie-80gb-20260913\?tab=architecture/);
  await expect(page.locator('.arch-section-2 [data-fact="tensor_count"] strong')).toHaveText("456");
  await page.getByRole("tab", { name: "Specifications" }).click();
  await expect(page.getByRole("heading", { name: "Compute capabilities" })).toBeVisible();
});

test("desktop architecture four aligned rows and chip table placement", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "Desktop composition regression");
  for (const width of [1280, 1440, 1920]) {
    await page.setViewportSize({ width, height: 1080 });
    await page.goto("/hardware/nvidia-a100-pcie-80gb-20260913?tab=architecture");
    await expect(page.locator(".arch-row")).toHaveCount(4);
    const layout = await page.evaluate(() => {
      const box = (el: Element) => {
        const r = el.getBoundingClientRect();
        return { x: r.x, y: r.y, width: r.width, height: r.height, bottom: r.bottom };
      };
      return {
        rows: Array.from(document.querySelectorAll(".arch-row")).map(row =>
          Array.from(row.querySelectorAll(".arch-section")).map(el => ({ ...box(el), id: el.id }))),
        metrics: box(document.querySelector(".arch-chip-resources")!),
        table: box(document.querySelector(".arch-chip-table")!),
        education: box(document.querySelector(".arch-legacy")!),
        overview: box(document.querySelector(".arch-comparison")!),
        overflow: document.documentElement.scrollWidth > innerWidth + 1,
      };
    });
    expect(layout.rows.map(r => r.map(s => s.id))).toEqual([
      ["architecture-1", "architecture-2"], ["architecture-4", "architecture-3"],
      ["architecture-6", "architecture-5"], ["architecture-7", "architecture-8"],
    ]);
    for (const [left, right] of layout.rows) {
      expect(Math.abs(left.y - right.y)).toBeLessThan(1);
      expect(Math.abs(left.height - right.height)).toBeLessThan(1);
      expect(right.width / left.width).toBeCloseTo(2, 1);
    }
    expect(layout.table.x).toBeGreaterThan(layout.metrics.x + layout.metrics.width);
    expect(Math.abs(layout.metrics.y - layout.table.y)).toBeLessThan(1);
    expect(layout.overview.y).toBeGreaterThanOrEqual(layout.education.bottom);
    expect(layout.overflow).toBe(false);
    await page.screenshot({ path: `${evidence}/a100-desktop-${width}.png`, fullPage: true });
    if (width === 1440) {
      await page.locator(".arch-row").first().screenshot({ path: `${evidence}/a100-first-row.png` });
      await page.locator(".arch-row").nth(1).screenshot({ path: `${evidence}/a100-second-row.png` });
    }
  }
});

test("architecture API error recovery and NPU no fabricated SM", async ({ page }) => {
  const route = "**/api/hardware/nvidia-b200-sxm-180gb-20260913/architecture";
  await page.route(route, route => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Architecture source unavailable" }) }));
  await page.goto("/hardware/nvidia-b200-sxm-180gb-20260913?tab=architecture");
  await expect(page.getByRole("alert")).toContainText("Architecture source unavailable");
  await page.unroute(route);
  await page.getByRole("button", { name: "Retry architecture" }).click();
  await expect(page.locator(".arch-section")).toHaveCount(8);
  await page.goto("/hardware/huawei-910c-module-128gb-20260913?tab=architecture");
  await expect(page.locator(".arch-section")).toHaveCount(8);
  await expect(page.locator(".arch-sm-facts .arch-fact")).toHaveCount(0);
  await expect(page.locator(".arch-memory")).toContainText("缺少片上微架构资料");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
});
