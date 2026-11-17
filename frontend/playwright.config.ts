import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
export default defineConfig({
  testDir: "./tests",
  outputDir: process.env.PLANNER_E2E_EVIDENCE
    ? `${process.env.PLANNER_E2E_EVIDENCE}/artifacts` : "./test-results",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  reporter: [
    ["list"],
    [
      "json",
      {
        outputFile: `${process.env.PLANNER_E2E_EVIDENCE ?? "./test-results"}/playwright.json`,
      },
    ],
  ],
  use: {
    baseURL: process.env.PLANNER_E2E_BASE_URL ?? "http://127.0.0.1:8675",
    channel: process.env.PLAYWRIGHT_EXECUTABLE_PATH ? undefined : process.env.PLAYWRIGHT_CHANNEL ?? "chrome",
    launchOptions: process.env.PLAYWRIGHT_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH } : undefined,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 1000 },
      },
    },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], defaultBrowserType: "chromium" },
    },
  ],
  webServer: process.env.PLANNER_E2E_BASE_URL ? undefined : {
    command:
      "../.venv/bin/python -m uvicorn app.main:app --app-dir ../backend --host 127.0.0.1 --port 8675",
    url: "http://127.0.0.1:8675/api/health",
    env: { DATABASE_URL: process.env.PLANNER_E2E_DATABASE_URL ?? `sqlite:///${root}/data/e2e.db` },
    reuseExistingServer: false,
    timeout: 30000,
  },
});
