import { expect, test } from "@playwright/test";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-13-hardware-center/evidence";
const h100 = "nvidia-h100-sxm-80gb-20260913";

test("hardware filters, details, roofline, export and workload handoff", async ({ page }, info) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  await page.goto("/hardware");
  await expect(page.getByText("11 GPU SKUs")).toBeVisible();
  await page.screenshot({ path: `${evidence}/hw-list-${info.project.name}.png`, fullPage: true });
  await page.getByLabel("Search GPUs").fill("no-match");
  await expect(page.getByText("No matching GPUs.")).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(page.getByLabel("Search GPUs")).toHaveValue("");
  await expect(page.getByText("11 GPU SKUs")).toBeVisible();
  await page.getByLabel("Vendor", { exact: true }).selectOption("Huawei");
  await expect(page.getByText("2 GPU SKUs")).toBeVisible();
  await page.getByLabel("Vendor", { exact: true }).selectOption("");
  await expect(page.getByText("11 GPU SKUs")).toBeVisible();
  await page.getByLabel("Form factor").selectOption("PCIe");
  await expect(page.getByText("4 GPU SKUs")).toBeVisible();
  await page.getByLabel("Form factor").selectOption("");
  await expect(page.getByText("11 GPU SKUs")).toBeVisible();
  await page.getByRole("link", { name: "H100 SXM 80GB", exact: true }).click();
  await expect(page.getByRole("heading", { name: "H100 SXM 80GB", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Roofline", exact: true }).click();
  await expect(page.getByRole("img", { name: "BF16 theoretical roofline" })).toBeVisible();
  await expect.poll(() => page.locator(".hw-plot").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBeTruthy();
  await page.getByLabel("Arithmetic intensity").fill("10000");
  await expect(page.locator(".facts dd").filter({ hasText: /^compute$/ })).toBeVisible();
  await page.getByLabel("Precision", { exact: true }).selectOption("fp8");
  await expect(page.getByRole("img", { name: "FP8 theoretical roofline" })).toBeVisible();
  await page.getByLabel("Structured sparsity").check();
  await expect(page.getByText("3,958 TFLOPS", { exact: true }).first()).toBeVisible();
  await page.getByLabel("Arithmetic intensity").fill("0");
  await expect(page.getByRole("alert")).toContainText("greater than 0");
  await page.getByLabel("Arithmetic intensity").fill("64");
  await page.getByLabel("Structured sparsity").uncheck();
  await page.getByLabel("Precision", { exact: true }).selectOption("bf16");
  await expect(page.getByRole("img", { name: "BF16 theoretical roofline" })).toBeVisible();
  await page.screenshot({ path: `${evidence}/hw-roofline-${info.project.name}.png`, fullPage: true });
  await page.getByRole("tab", { name: "Sources & caveats" }).click();
  await expect(page.getByText(/CSV 989 为四舍五入/)).toBeVisible();
  await expect(page.getByText("Content SHA256", { exact: true })).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByTitle("Export hardware JSON").click();
  expect((await download).suggestedFilename()).toBe(`${h100}.json`);
  await page.getByRole("link", { name: "Create workload" }).click();
  await expect(page.getByLabel("Hardware", { exact: true })).toHaveValue(h100);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
  expect(errors).toEqual([]);
});

test("hardware unavailable precision, reference-only data and error recovery", async ({ page }, info) => {
  await page.goto("/hardware/nvidia-a100-pcie-80gb-20260913?tab=roofline");
  await page.getByLabel("Precision", { exact: true }).selectOption("fp8");
  await expect(page.getByRole("alert")).toContainText("unsupported or unverified");
  await page.goto("/hardware/huawei-910c-module-128gb-20260913?tab=roofline");
  await expect(page.getByRole("alert")).toContainText("unverified");
  await expect(page.getByRole("link", { name: "Create workload" })).toHaveCount(0);
  await page.goto("/hardware/missing");
  await expect(page.getByRole("alert")).toContainText("not found");
  await page.getByRole("link", { name: "GPU Catalog", exact: true }).click();
  await expect(page.getByText("11 GPU SKUs")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
  await page.screenshot({ path: `${evidence}/hw-recovery-${info.project.name}.png`, fullPage: true });
});

test("hardware architecture education and packaging diagram", async ({ page }, info) => {
  await page.goto(`/hardware/${h100}?tab=architecture`);
  await expect(page.getByRole("heading", { name: "Hopper：FP8 与 Transformer Engine" })).toBeVisible();
  await expect(page.getByRole("figure", { name: "Accelerator packaging diagram" })).toBeVisible();
  await expect(page.getByText("HGX / OEM 基板", { exact: true })).toBeVisible();
  await expect(page.locator(".hw-form-factor img")).toHaveCount(0);
  await page.screenshot({ path: `${evidence}/hw-architecture-${info.project.name}.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
});
