import { test, expect } from "@playwright/test";
import type { APIRequestContext } from "@playwright/test";
import { initialInput } from "../src/features/workload-designer/state";

async function prepared(request: APIRequestContext, suffix: string) {
  const models = await (await request.get("/api/models")).json();
  const model = models.find((m: { repo_id: string }) => m.repo_id === "Qwen/Qwen2.5-0.5B");
  const input = initialInput(model.id, "nvidia-h100-sxm-80gb-20260913");
  Object.assign(input.load, { prompt_tokens: 5, output_tokens: 3, concurrency: 2, request_count: 2, request_rate: null, arrival: null });
  input.mode = "offline"; input.runtime.max_batch_sequences = 2;
  input.parallelism.mode = "manual";
  input.parallelism.manual_candidate = { tp: 2, pp: 2, ep: 1, dp: 1, boundaries: [12] };
  const result = await request.post("/api/workloads/designs", {
    data: { name: `History ${suffix} ${Date.now()}`, input }, headers: { "Idempotency-Key": crypto.randomUUID() },
  });
  expect(result.status()).toBe(201);
  return result.json();
}

test("revision recovery, CAS conflict, cross-run compare and placement/communication", async ({ page, request }, info) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  const design = await prepared(request, info.project.name);
  await page.goto(`/workloads/inference/${design.id}`);
  if (info.project.name === "desktop") await page.setViewportSize({ width: 1280, height: 768 });
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  const firstRun = new URL(page.url()).searchParams.get("run")!;
  await page.locator(".wc-candidates input").first().check();
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Placement", exact: true }).click();
  await page.locator(".wc-gpus button").filter({ hasText: "R2" }).click();
  await expect(page.locator(".wc-inspector")).toContainText("此 GPU 不持有所选层");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByLabel("Layer", { exact: true }).selectOption("12");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.locator(".wc-inspector")).not.toContainText("此 GPU 不持有所选层");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Communication", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await expect(page.getByLabel("Expert Mode")).toHaveCount(0);
  await page.locator(".wc-network-edge-label").first().click();
  await expect(page.getByLabel("Link details", { exact: true })).toContainText("Bandwidth / GPU");
  await page.getByLabel("Link details", { exact: true }).getByText("Timing & paths", { exact: true }).click();
  await expect(page.getByLabel("Link details", { exact: true })).toContainText("gpu:");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  await page.getByLabel("Batch Size", { exact: true }).fill("1");
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get("run")).not.toBe(firstRun);
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  const secondRun = new URL(page.url()).searchParams.get("run")!;
  await page.locator(".wc-candidates input").first().check();
  await page.getByRole("link", { name: "Compare (2)", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Input Differences" })).toBeVisible();
  await expect(page.locator(".wc-compare")).toContainText("runtime.max_batch_sequences");
  await page.locator(".wc-compare thead a").first().click();
  await expect(page).toHaveURL(new RegExp(`/workloads/inference/${design.id}\\?`));
  await expect(page.locator(".wc-run-status")).toContainText("completed");
  await page.getByText("Revision History & Sources", { exact: true }).click();
  await page.getByLabel("Revision", { exact: true }).selectOption("1");
  if (info.project.name === "mobile") await page.getByRole("button", { name: "Parameters", exact: true }).click();
  await expect(page.getByLabel("Batch Size", { exact: true })).toHaveValue("2");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.locator(".wc-header")).toContainText("Revision 3");
  const head = await (await request.get(`/api/workloads/designs/${design.id}`)).json();
  head.revision.input.runtime.max_batch_sequences = 1;
  expect((await request.post(`/api/workloads/designs/${design.id}/revisions`, {
    headers: { "If-Match": head.etag }, data: { input: head.revision.input },
  })).status()).toBe(201);
  await page.getByLabel("Prompt Tokens", { exact: true }).fill("6");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.locator(".wc-error")).toContainText("REVISION_CONFLICT");
  await page.locator(".wc-error").getByRole("button", { name: "Reload Head", exact: true }).click();
  await expect(page.getByLabel("Batch Size", { exact: true })).toHaveValue("1");
  await expect(page.getByLabel("Prompt Tokens", { exact: true })).toHaveValue("5");
  await page.getByLabel("Run History", { exact: true }).selectOption(secondRun);
  await page.getByRole("button", { name: "Use Run Input", exact: true }).click();
  await expect(page.getByText(/Stale Result/)).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(errors).toEqual([]);
});

test("cancel and retry retain frozen input", async ({ page, request }, info) => {
  test.setTimeout(90000);
  const design = await prepared(request, `cancel ${info.project.name}`);
  await page.goto(`/workloads/inference/${design.id}`);
  await page.getByRole("button", { name: "Evaluate", exact: true }).click();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.locator(".wc-run-status")).toContainText("cancelled", { timeout: 15000 });
  const cancelledId = new URL(page.url()).searchParams.get("run");
  const cancelled = await (await request.get(`/api/workloads/runs/${cancelledId}`)).json();
  await page.getByRole("button", { name: "Retry Frozen Input", exact: true }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get("run")).not.toBe(cancelledId);
  await expect(page.locator(".wc-run-status")).toContainText("completed", { timeout: 45000 });
  const retryId = new URL(page.url()).searchParams.get("run");
  const retried = await (await request.get(`/api/workloads/runs/${retryId}`)).json();
  expect(retried.input_hash).toBe(cancelled.input_hash);
  expect((await (await request.get(`/api/workloads/runs/${cancelledId}`)).json()).status).toBe("cancelled");
});
