import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./client";
import { compact, gib, number } from "../utils/format";

afterEach(() => vi.unstubAllGlobals());
describe("API boundary", () => {
  it("returns JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response('{"status":"ok"}')),
    );
    expect(await request("/health")).toEqual({ status: "ok" });
  });
  it("normalizes validation details", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              detail: [
                { loc: ["body", "batch_size"], msg: "Must be positive" },
              ],
            }),
            { status: 422 },
          ),
        ),
    );
    await expect(request("/workloads")).rejects.toThrow(
      "batch_size: Must be positive",
    );
  });
  it("preserves HTTP error status", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response('{"detail":"Model missing"}', { status: 404 }),
        ),
    );
    await expect(request("/models/unknown")).rejects.toEqual(
      new ApiError("Model missing", 404),
    );
  });
  it("handles non-JSON failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response("Gateway failure", { status: 502 })),
    );
    await expect(request("/models")).rejects.toThrow("Request failed (502)");
  });
});
describe("display units", () => {
  it("labels binary memory explicitly", () =>
    expect(gib(2 ** 30)).toBe("1 GiB"));
  it("formats SI parameter counts", () =>
    expect(compact(1543714304)).toBe("1.54B"));
  it("preserves exact parameter totals", () =>
    expect(number(494032768)).toBe("494,032,768"));
});
