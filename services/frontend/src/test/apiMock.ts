// Test helper: stub `fetch` with per-route handlers keyed by "METHOD /path"
// (path relative to the /api/v1 base), so components exercise the real API
// client against canned responses.

import { vi } from "vitest";

type Handler = (init: RequestInit | undefined) => Response;

export function mockFetch(routes: Record<string, Handler>): void {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    // Match on the path only; query strings (e.g. pagination) don't affect routing.
    const path = url.replace("/api/v1", "").split("?")[0];
    const handler = routes[`${method} ${path}`];
    if (!handler) {
      return new Response(JSON.stringify({ detail: "not mocked", code: "NOT_FOUND" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }
    return handler(init);
  });
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function noContent(): Response {
  return new Response(null, { status: 204 });
}

export function unauthorized(): Response {
  return new Response(JSON.stringify({ detail: "Not authenticated.", code: "UNAUTHENTICATED" }), {
    status: 401,
    headers: { "content-type": "application/json" },
  });
}

/** Build one SSE frame (matching the backend's `event:`/`data:` shape). */
export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** A streaming `text/event-stream` response whose body emits the given frames. */
export function sse(frames: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}
