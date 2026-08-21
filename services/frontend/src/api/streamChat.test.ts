// Tests for the chat SSE client: it parses the backend's ordered frames and
// dispatches them to the handlers, including the trace id on the terminal frame.

import { afterEach, describe, expect, it, vi } from "vitest";

import { mockFetch, sse, sseFrame } from "../test/apiMock";
import { streamChat } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("streamChat", () => {
  it("dispatches frames in order and surfaces the trace id on done", async () => {
    mockFetch({
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("token", { text: "Hi " }),
          sseFrame("token", { text: "there." }),
          sseFrame("done", { answer: "Hi there.", trace_id: "trace-1" }),
        ]),
    });

    const tokens: string[] = [];
    let conversationId: string | undefined;
    let doneAnswer: string | undefined;
    let doneTrace: string | null | undefined;

    await streamChat(
      { message: "hi" },
      {
        onConversation: (id) => (conversationId = id),
        onToken: (t) => tokens.push(t),
        onDone: (answer, traceId) => {
          doneAnswer = answer;
          doneTrace = traceId;
        },
      },
    );

    expect(conversationId).toBe("conv-1");
    expect(tokens.join("")).toBe("Hi there.");
    expect(doneAnswer).toBe("Hi there.");
    expect(doneTrace).toBe("trace-1");
  });

  it("passes null trace id when the done frame omits it", async () => {
    mockFetch({
      "POST /chat/stream": () => sse([sseFrame("done", { answer: "ok" })]),
    });

    let doneTrace: string | null | undefined = "unset";
    await streamChat({ message: "hi" }, { onDone: (_a, traceId) => (doneTrace = traceId) });

    expect(doneTrace).toBeNull();
  });
});
