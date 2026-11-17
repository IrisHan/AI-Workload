import { test, expect } from "@playwright/test";

const evidence = process.env.PLANNER_E2E_EVIDENCE ?? "../docs/reports/2026-09-15-moe-ffn-structure/evidence";
test.setTimeout(90000);

async function ensureImported(page: import("@playwright/test").Page, request: import("@playwright/test").APIRequestContext, name: string, repo: string) {
  const models = await (await request.get("/api/models")).json() as { id: string; name: string }[];
  const existing = models.find(item => item.name.includes(name));
  if (existing) return existing;
  await page.goto("/models/import");
  await page.getByLabel("Repository ID or URL", { exact: true }).fill(`https://huggingface.co/${repo}`);
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible({ timeout: 40000 });
  const after = await (await request.get("/api/models")).json() as { id: string; name: string }[];
  const model = after.find(item => item.name.includes(name));
  expect(model, `imported model ${name}`).toBeTruthy();
  return model!;
}

async function openStructure(page: import("@playwright/test").Page, request: import("@playwright/test").APIRequestContext, name: string, repo: string) {
  const model = await ensureImported(page, request, name, repo);
  const snapshot = await (await request.post(`/api/models/${model.id}/structure`)).json();
  await page.goto(`/models/${model.id}?tab=structure&snapshot=${snapshot.id}`);
  const studio = page.getByRole("region", { name: "Spec Studio model workspace" });
  await expect(studio.locator(".react-flow__node").first()).toBeVisible({ timeout: 20000 });
  return { studio, snapshot, model };
}

test("Dense Qwen FFN stays SwiGLU without router", async ({ page, request }, info) => {
  const { studio } = await openStructure(page, request, "Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B");
  await expect(page.getByTestId("ffn-recipe")).toContainText("Dense SwiGLU");
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "gate_proj" })).toHaveCount(1);
  await expect(studio.locator(".react-flow__node").filter({ hasText: "router" })).toHaveCount(0);
  await expect(studio.locator(".module-table tbody tr")).toHaveCount(3);
  if (info.project.name === "desktop")
    await page.screenshot({ path: `${evidence}/dense-qwen25-ffn-${info.project.name}.png`, fullPage: false });
});

test("Coder MoE FFN shows routed experts without shared", async ({ page, request }, info) => {
  const { studio } = await openStructure(page, request, "Qwen3-Coder-480B", "Qwen/Qwen3-Coder-480B-A35B-Instruct");
  await expect(page.getByTestId("ffn-recipe")).toContainText("MoE");
  await expect(page.getByTestId("ffn-recipe")).toContainText("×160");
  await expect(page.getByTestId("ffn-recipe")).toContainText("no shared");
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "router" }).first()).toBeVisible();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "×160" }).first()).toBeVisible();
  await expect(studio.locator(".module-table")).toContainText("routed_expert.gate_proj");
  await expect(studio.locator(".module-table")).not.toContainText("shared_expert.gate_proj");
  if (info.project.name === "desktop")
    await page.screenshot({ path: `${evidence}/coder-moe-ffn-${info.project.name}.png`, fullPage: false });
});

test("Next hybrid MoE FFN shows shared expert and shared gate", async ({ page, request }, info) => {
  const { studio } = await openStructure(page, request, "Qwen3-Next-80B", "Qwen/Qwen3-Next-80B-A3B-Instruct");
  await expect(page.getByTestId("ffn-recipe")).toContainText("shared_expert_gate");
  await expect(page.getByTestId("ffn-recipe")).toContainText("×512");
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "shared_expert.gate_proj" }).first()).toBeVisible();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "shared_expert_gate" }).first()).toBeVisible();
  if (info.project.name === "desktop")
    await page.screenshot({ path: `${evidence}/next-moe-ffn-${info.project.name}.png`, fullPage: false });
});

test("DeepSeek V4 FFN still shows routed and shared experts", async ({ page, request }, info) => {
  const { studio } = await openStructure(page, request, "DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Flash-Base");
  await expect(page.getByTestId("ffn-recipe")).toContainText("MoE");
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "routed_expert.gate_proj" }).first()).toBeVisible();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "shared_expert.gate_proj" }).first()).toBeVisible();
  if (info.project.name === "desktop")
    await page.screenshot({ path: `${evidence}/v4-moe-ffn-${info.project.name}.png`, fullPage: false });
});

test("DeepSeek MoE prefix defaults to the MoE group", async ({ page, request }, info) => {
  const { studio } = await openStructure(page, request, "deepseek-moe-16b-base", "deepseek-ai/deepseek-moe-16b-base");
  await expect(page.getByTestId("ffn-recipe")).toContainText("MoE");
  await expect(page.getByTestId("ffn-recipe")).toContainText("routed");
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "router" }).first()).toBeVisible();
  await expect(studio.locator(".module-table")).toContainText("routed_expert.gate_proj");
  if (info.project.name === "desktop")
    await page.screenshot({ path: `${evidence}/deepseek-moe16-ffn-${info.project.name}.png`, fullPage: false });
});

test("DeepSeek V3.2 MoE FFN shows router and routed experts", async ({ page, request }, info) => {
  const { studio } = await openStructure(page, request, "DeepSeek-V3.2-Exp-Base", "deepseek-ai/DeepSeek-V3.2-Exp-Base");
  await expect(page.getByTestId("morphology-tags")).toContainText("DSA");
  await expect(page.getByTestId("ffn-recipe")).toContainText("MoE");
  await studio.getByRole("button", { name: "FFN", exact: true }).click();
  await expect(studio.locator(".react-flow__node").filter({ hasText: "router" }).first()).toBeVisible();
  await expect(studio.locator(".module-table")).toContainText("routed_expert.gate_proj");
  if (info.project.name === "desktop")
    await page.screenshot({ path: `${evidence}/v32-moe-ffn-${info.project.name}.png`, fullPage: false });
});
