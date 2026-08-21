import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Conversation } from "../api/types";
import { json, mockFetch, noContent, sse, sseFrame } from "../test/apiMock";
import { Chat } from "./Chat";

// Chat.tsx feature-detects recording support with a module-load-time constant,
// so the fake MediaRecorder/getUserMedia must exist *before* Chat is imported —
// vi.hoisted runs its callback above this file's imports for exactly that reason.
const { getUserMediaMock } = vi.hoisted(() => {
  class FakeMediaRecorder {
    static isTypeSupported(type: string): boolean {
      return type === "audio/webm";
    }
    // Real browsers report mimeType with a codec parameter attached — the fake
    // mirrors that so the test guards against sending that raw string (rather
    // than the bare "audio/webm" the backend's allowlist expects) on to the API.
    mimeType = "audio/webm;codecs=opus";
    state: "inactive" | "recording" = "inactive";
    ondataavailable: ((event: { data: Blob }) => void) | null = null;
    onstop: (() => void) | null = null;
    start(): void {
      this.state = "recording";
    }
    stop(): void {
      this.state = "inactive";
      // Spec order: a final dataavailable (with the buffered audio) fires
      // before stop.
      this.ondataavailable?.({
        data: new Blob([new Uint8Array([1, 2, 3])], { type: this.mimeType }),
      });
      this.onstop?.();
    }
  }
  const getUserMediaMock = vi.fn();
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  Object.defineProperty(globalThis.navigator, "mediaDevices", {
    value: { getUserMedia: getUserMediaMock },
    writable: true,
    configurable: true,
  });
  return { getUserMediaMock };
});

afterEach(() => {
  vi.restoreAllMocks();
});

beforeEach(() => {
  // restoreAllMocks (above) resets getUserMediaMock's implementation too, so
  // the happy-path default is re-armed before every test.
  getUserMediaMock.mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] });
});

function conversationPage(items: Conversation[] = []) {
  return { items, total: items.length, page: 1, page_size: 50 };
}

function conversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "conv-1",
    title: "The Odyssey",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("Chat", () => {
  it("streams a turn: tool step, answer tokens, then the final answer", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("node_status", { node: "plan", description: "Planning how to respond..." }),
          sseFrame("tool_call", { name: "retrieve_chunks", args: { query: "narrator" }, id: "c1" }),
          sseFrame("tool_result", { name: "retrieve_chunks", content: "[1] Odysseus…", id: "c1" }),
          sseFrame("token", { text: "Odysseus " }),
          sseFrame("token", { text: "narrates." }),
          sseFrame("done", { answer: "Odysseus narrates." }),
        ]),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), {
      target: { value: "who narrates?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    // The user's prompt shows immediately; the assistant's final answer arrives
    // after the stream completes — the interleaved node_status frame doesn't
    // interfere with the rest of the turn.
    expect(await screen.findByText(/who narrates\?/)).toBeInTheDocument();
    expect(await screen.findByText("Odysseus narrates.")).toBeInTheDocument();
    // The tool step the agent took is surfaced.
    expect(screen.getByText(/retrieve_chunks/)).toBeInTheDocument();
    expect(screen.getByText(/1 tool step/i)).toBeInTheDocument();
  });

  it("shows a live progress description while streaming with no content yet", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("node_status", { node: "plan", description: "Planning how to respond..." }),
          // Nothing else follows: the stream ends here so the turn stays
          // mid-flight, letting the assertion below observe the live status
          // instead of racing a final answer that would replace it.
        ]),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), {
      target: { value: "who narrates?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    expect(await screen.findByText("Planning how to respond...")).toBeInTheDocument();
  });

  it("renders a guardrail refusal as a single blocked reply", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("blocked", { reason: "I only help with your reading." }),
        ]),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), {
      target: { value: "write me a poem about taxes" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/only help with your reading/i);
  });

  it("loads a past conversation's transcript when selected", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage([conversation()])),
      "GET /conversations/conv-1/messages": () =>
        json({
          items: [
            {
              id: "m1",
              role: "user",
              content: "who narrates?",
              tool_calls: null,
              created_at: "2026-01-01T00:00:00Z",
            },
            {
              id: "m2",
              role: "assistant",
              content: "Odysseus does.",
              tool_calls: { steps: [{ name: "retrieve_chunks", args: {}, result: "[1] …" }] },
              created_at: "2026-01-01T00:00:01Z",
            },
          ],
          total: 2,
          page: 1,
          page_size: 100,
        }),
    });
    render(<Chat />);

    fireEvent.click(await screen.findByRole("button", { name: /The Odyssey/i }));

    expect(await screen.findByText("Odysseus does.")).toBeInTheDocument();
    expect(screen.getByText(/who narrates\?/)).toBeInTheDocument();
  });

  it("deletes a conversation and removes it from the picker", async () => {
    let deleted = false;
    mockFetch({
      "GET /conversations": () =>
        json(conversationPage(deleted ? [] : [conversation()])),
      "DELETE /conversations/conv-1": () => {
        deleted = true;
        return noContent();
      },
    });
    render(<Chat />);
    await screen.findByRole("button", { name: /The Odyssey/i });

    fireEvent.click(screen.getByRole("button", { name: /delete conversation/i }));

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /The Odyssey/i })).not.toBeInTheDocument(),
    );
  });

  it("clears the transcript when the active conversation is deleted", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage([conversation()])),
      "GET /conversations/conv-1/messages": () =>
        json({
          items: [
            {
              id: "m1",
              role: "user",
              content: "who narrates?",
              tool_calls: null,
              created_at: "2026-01-01T00:00:00Z",
            },
          ],
          total: 1,
          page: 1,
          page_size: 100,
        }),
      "DELETE /conversations/conv-1": () => noContent(),
    });
    render(<Chat />);

    fireEvent.click(await screen.findByRole("button", { name: /The Odyssey/i }));
    expect(await screen.findByText(/who narrates\?/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /delete conversation/i }));

    await waitFor(() => expect(screen.queryByText(/who narrates\?/)).not.toBeInTheDocument());
  });

  it("shows an error when deleting a conversation fails", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage([conversation()])),
      "DELETE /conversations/conv-1": () =>
        json({ detail: "not found", code: "NOT_FOUND" }, 404),
    });
    render(<Chat />);
    await screen.findByRole("button", { name: /The Odyssey/i });

    fireEvent.click(screen.getByRole("button", { name: /delete conversation/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/couldn't delete/i);
  });

  it("resolves a page-range-confirmation interrupt (FR-4.6)", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("interrupt", {
            kind: "page_range_confirm",
            document_id: "doc-1",
            document_title: "The Odyssey",
            proposal: { page_start: 21, page_end: 50, proposal_reason: "pages read since the last summary" },
          }),
        ]),
      "POST /chat/conv-1/resume": () =>
        json({
          conversation_id: "conv-1",
          answer: "Saved a summary for pages 21-50.",
          blocked: false,
          tool_steps: [],
        }),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), {
      target: { value: "what's my progress?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    expect(await screen.findByLabelText(/confirm summary range/i)).toBeInTheDocument();
    expect(screen.getByText(/The Odyssey/)).toBeInTheDocument();
    // Sending a new message is disabled while a decision is pending.
    expect(screen.getByLabelText(/message/i)).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /save summary/i }));

    expect(await screen.findByText("Saved a summary for pages 21-50.")).toBeInTheDocument();
    expect(screen.queryByLabelText(/confirm summary range/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/message/i)).not.toBeDisabled();
  });

  it("resolves an ask-pages-read interrupt (FR-4.7)", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("interrupt", {
            kind: "ask_pages_read",
            document_id: "doc-1",
            document_title: "The Odyssey",
            reason: "No reading position is tracked yet.",
          }),
        ]),
      "POST /chat/conv-1/resume": () =>
        json({
          conversation_id: "conv-1",
          answer: "They faced trials at sea.",
          blocked: false,
          tool_steps: [],
        }),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), { target: { value: "catch me up" } });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    const pageInput = await screen.findByLabelText(/up to which page/i);
    fireEvent.change(pageInput, { target: { value: "50" } });
    fireEvent.click(screen.getByRole("button", { name: /^submit$/i }));

    expect(await screen.findByText("They faced trials at sea.")).toBeInTheDocument();
  });

  it("resolves a tool-approval interrupt by approving", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("interrupt", {
            kind: "tool_approval",
            tool_call: { name: "web_search", args: { query: "reviews" }, id: "c1" },
            reason: "'web_search' reaches beyond your stored data and needs your approval.",
          }),
        ]),
      "POST /chat/conv-1/resume": () =>
        json({
          conversation_id: "conv-1",
          answer: "Here's what I found.",
          blocked: false,
          tool_steps: [{ name: "web_search", args: { query: "reviews" }, result: "…" }],
        }),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), {
      target: { value: "search for reviews" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    expect(await screen.findByLabelText(/approval needed/i)).toBeInTheDocument();
    expect(screen.getByText("web_search")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^approve$/i }));

    expect(await screen.findByText("Here's what I found.")).toBeInTheDocument();
  });

  it("resolves a spoiler-warning interrupt (FR-18.4)", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("interrupt", {
            kind: "spoiler_warning",
            document_id: "doc-1",
            document_title: "The Odyssey",
            current_page: 50,
            reason: "reveals who Odysseus fights in the end",
          }),
        ]),
      "POST /chat/conv-1/resume": () =>
        json({
          conversation_id: "conv-1",
          answer: "Odysseus defeats the suitors.",
          blocked: false,
          tool_steps: [],
        }),
    });
    render(<Chat />);

    fireEvent.change(screen.getByLabelText(/message/i), {
      target: { value: "what happens at the end?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    expect(await screen.findByLabelText(/spoiler warning/i)).toBeInTheDocument();
    expect(screen.getByText(/reveals who Odysseus fights in the end/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /show me anyway/i }));

    expect(await screen.findByText("Odysseus defeats the suitors.")).toBeInTheDocument();
    expect(screen.queryByLabelText(/spoiler warning/i)).not.toBeInTheDocument();
  });

  it("shows a real thumbnail preview for a just-sent image attachment", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": () =>
        sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("done", { answer: "Nice cover!" }),
        ]),
    });
    render(<Chat />);

    const file = new File([new Uint8Array([1, 2, 3])], "cover.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText(/attach audio or image/i), {
      target: { files: [file] },
    });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    const previews = await screen.findByTestId("attachment-previews");
    expect(previews.querySelector("img")).toHaveAttribute("alt", "cover.png");
  });

  it("renders a reloaded historical attachment note as an icon badge, not raw brackets", async () => {
    mockFetch({
      "GET /conversations": () => json(conversationPage([conversation()])),
      "GET /conversations/conv-1/messages": () =>
        json({
          items: [
            {
              id: "m1",
              role: "user",
              content: "what's on this cover?\n\n[1 image attachment]",
              tool_calls: null,
              created_at: "2026-01-01T00:00:00Z",
            },
          ],
          total: 1,
          page: 1,
          page_size: 100,
        }),
    });
    render(<Chat />);

    fireEvent.click(await screen.findByRole("button", { name: /The Odyssey/i }));

    expect(await screen.findByText("what's on this cover?")).toBeInTheDocument();
    expect(screen.queryByText(/\[1 image attachment\]/)).not.toBeInTheDocument();
    expect(screen.getByTestId("attachment-badges")).toHaveTextContent(/1 image/i);
  });

  it("keeps the compose controls usable and shows attachment count", async () => {
    mockFetch({ "GET /conversations": () => json(conversationPage()) });
    render(<Chat />);

    const file = new File([new Uint8Array([1, 2, 3])], "note.wav", { type: "audio/wav" });
    fireEvent.change(screen.getByLabelText(/attach audio or image/i), {
      target: { files: [file] },
    });

    await waitFor(() =>
      expect(screen.getByTestId("attachment-count")).toHaveTextContent(/1 attached/i),
    );
  });

  it("records a voice note from the mic and attaches it on stop", async () => {
    mockFetch({ "GET /conversations": () => json(conversationPage()) });
    render(<Chat />);

    fireEvent.click(screen.getByRole("button", { name: /record/i }));
    await waitFor(() => expect(getUserMediaMock).toHaveBeenCalledWith({ audio: true }));
    const stopButton = await screen.findByRole("button", { name: /stop/i });

    fireEvent.click(stopButton);

    await waitFor(() =>
      expect(screen.getByTestId("attachment-count")).toHaveTextContent(/1 attached/i),
    );
    // Stopping restores the "Record" control rather than leaving "Stop" showing.
    expect(screen.getByRole("button", { name: /record/i })).toBeInTheDocument();
  });

  it("sends the recorded clip with a bare mime type the backend's allowlist accepts", async () => {
    let sentPart: { kind: string; mime_type: string; data: string } | undefined;
    mockFetch({
      "GET /conversations": () => json(conversationPage()),
      "POST /chat/stream": (init) => {
        const body = JSON.parse(init?.body as string);
        sentPart = body.parts?.[0];
        return sse([
          sseFrame("conversation", { conversation_id: "conv-1" }),
          sseFrame("done", { answer: "Got it." }),
        ]);
      },
    });
    render(<Chat />);

    fireEvent.click(screen.getByRole("button", { name: /record/i }));
    const stopButton = await screen.findByRole("button", { name: /stop/i });
    fireEvent.click(stopButton);
    await waitFor(() =>
      expect(screen.getByTestId("attachment-count")).toHaveTextContent(/1 attached/i),
    );

    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));

    await waitFor(() => expect(sentPart).toBeDefined());
    // Not "audio/webm;codecs=opus" (what the fake recorder reports) — the raw
    // codec-qualified string would fail the backend's exact-match allowlist.
    expect(sentPart?.kind).toBe("audio");
    expect(sentPart?.mime_type).toBe("audio/webm");
    expect(sentPart?.data).not.toHaveLength(0);
  });

  it("shows an error when microphone access is denied", async () => {
    getUserMediaMock.mockRejectedValueOnce(new Error("Permission denied"));
    mockFetch({ "GET /conversations": () => json(conversationPage()) });
    render(<Chat />);

    fireEvent.click(screen.getByRole("button", { name: /record/i }));

    expect(await screen.findByText(/microphone access/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /record/i })).toBeInTheDocument();
  });
});
