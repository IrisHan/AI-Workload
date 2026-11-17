import { test, expect } from "@playwright/test";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-10-bootstrap/evidence";
test("model, architecture and inference design handoff", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Model Registry" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Qwen2.5-0.5B", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: `${evidence}/models-${info.project.name}.png`,
    fullPage: true,
  });
  await page
    .getByRole("textbox", { name: "Search models" })
    .fill("not-a-model");
  await expect(page.getByText("No matching models.")).toBeVisible();
  await page.getByRole("textbox", { name: "Search models" }).fill("");
  await page.getByRole("link", { name: "Qwen2.5-0.5B", exact: true }).click();
  await expect(page.locator(".facts > div").filter({
    has: page.getByText("Parameter count", { exact: true }),
  }).locator("dd")).toHaveText("494,032,768");
  await page.getByRole("tab", { name: "Architecture", exact: true }).click();
  await expect(page.getByText("Q projection / layer")).toBeVisible();
  await page.getByRole("tab", { name: "Source", exact: true }).click();
  await expect(
    page.getByText("060db6499f32faf8b98477b0a26969ef7d8b9987", { exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Create workload" }).click();
  await expect(page.getByLabel("Model", { exact: true })).toHaveValue("421fab2cd2f37c38775c3977");
  await page.getByLabel("Hardware", { exact: true }).selectOption("nvidia-h100-sxm-80gb-20260913");
  await page
    .getByLabel("Design Name")
    .fill(`E2E ${info.project.name} ${Date.now()}`);
  await page
    .getByRole("button", { name: "Save", exact: true })
    .click();
  await expect(page.locator(".wc-header")).toContainText("Revision 1");
  await page.screenshot({
    path: `${evidence}/design-handoff-${info.project.name}.png`,
    fullPage: true,
  });
  await page.reload();
  await expect(page.locator(".wc-header")).toContainText("Revision 1");
  await expect(page.locator(".wc-design-name")).toHaveText(/^E2E /);
  await expect(page.getByLabel("Design Name")).toHaveCount(0);
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "Recent designs" })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  const dimensions = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    width: window.innerWidth,
  }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1);
  expect(errors).toEqual([]);
});

test("navigation, hardware provenance, unknown route and API errors", async ({
  page,
}, info) => {
  await page.goto("/hardware");
  await expect(
    page.getByRole("heading", { name: "Hardware Catalog" }),
  ).toBeVisible();
  await expect(page.getByText("989.5", { exact: true })).toBeVisible();
  await page.screenshot({
    path: `${evidence}/hardware-${info.project.name}.png`,
    fullPage: true,
  });
  if (info.project.name === "mobile")
    await page.getByTitle("Open navigation").click();
  await page
    .getByRole("link", { name: "Resource Planner", exact: true })
    .click();
  await expect(page.getByText("Not implemented in v0.1")).toBeVisible();
  await page.goto("/unknown");
  await expect(
    page.getByRole("heading", { name: "Page not found" }),
  ).toBeVisible();
  await page.goto("/models/missing");
  await expect(page.getByRole("alert").first()).toContainText("not found");
});

test("legacy routes and unavailable training stay explicit", async ({
  page, request,
}, info) => {
  expect((await request.post("/api/workloads", { data: { task_type: "training" } })).status()).toBe(410);
  await page.goto("/workloads");
  await expect(page.getByRole("button", { name: "Training", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Fine-tuning", exact: true })).toBeDisabled();
  await page.goto(`/workloads/removed-${info.project.name}`);
  await expect(page.getByRole("heading", { name: "Legacy Workload Removed" })).toBeVisible();
  await page.getByRole("link", { name: "New Inference Design" }).click();
  await expect(page.getByLabel("Design Name")).toBeVisible();
});
