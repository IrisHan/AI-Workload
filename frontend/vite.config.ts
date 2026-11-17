import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8666",
      "/docs": "http://127.0.0.1:8666",
      "/openapi.json": "http://127.0.0.1:8666",
    },
  },
  build: { sourcemap: true },
  test: { include: ["src/**/*.test.ts"] },
});
