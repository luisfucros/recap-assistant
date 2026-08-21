import { afterEach, describe, expect, it, vi } from "vitest";

import { apiFetch, setSessionExpiredHandler } from "./client";
import { json, mockFetch, noContent, unauthorized } from "../test/apiMock";

afterEach(() => {
  vi.restoreAllMocks();
  setSessionExpiredHandler(null);
});

describe("apiFetch session renewal", () => {
  it("silently refreshes an expired access token and retries the original request once", async () => {
    let progressCalls = 0;
    const refresh = vi.fn(() => noContent());
    mockFetch({
      "GET /progress": () => {
        progressCalls += 1;
        return progressCalls === 1 ? unauthorized() : json({ in_progress: [], completed: [] });
      },
      "POST /auth/refresh": refresh,
    });

    const res = await apiFetch("/progress");

    expect(refresh).toHaveBeenCalledTimes(1);
    expect(progressCalls).toBe(2);
    expect(res.status).toBe(200);
  });

  it("surfaces the original 401 and notifies the session-expired handler when refresh also fails", async () => {
    const onExpired = vi.fn();
    setSessionExpiredHandler(onExpired);
    mockFetch({
      "GET /progress": () => unauthorized(),
      "POST /auth/refresh": () => unauthorized(),
    });

    const res = await apiFetch("/progress");

    expect(res.status).toBe(401);
    expect(onExpired).toHaveBeenCalledTimes(1);
  });

  it("never attempts a refresh for the pre-session auth routes themselves", async () => {
    const refresh = vi.fn(() => unauthorized());
    mockFetch({
      "POST /auth/login": () => unauthorized(),
      "POST /auth/refresh": refresh,
    });

    const res = await apiFetch("/auth/login", { method: "POST", body: "{}" });

    expect(res.status).toBe(401);
    expect(refresh).not.toHaveBeenCalled();
  });

  it("dedupes concurrent refreshes into a single /auth/refresh call", async () => {
    const refresh = vi.fn(() => noContent());
    let aCalls = 0;
    let bCalls = 0;
    mockFetch({
      "GET /a": () => {
        aCalls += 1;
        return aCalls === 1 ? unauthorized() : json({ ok: true });
      },
      "GET /b": () => {
        bCalls += 1;
        return bCalls === 1 ? unauthorized() : json({ ok: true });
      },
      "POST /auth/refresh": refresh,
    });

    const [resA, resB] = await Promise.all([apiFetch("/a"), apiFetch("/b")]);

    expect(refresh).toHaveBeenCalledTimes(1);
    expect(resA.status).toBe(200);
    expect(resB.status).toBe(200);
  });
});
