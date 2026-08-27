// The reading assistant chat.
//
// A conversation picker plus a live transcript: the user types (and can attach a
// voice note or image, or record one live from the microphone, FR-19), the turn
// streams back token-by-token over SSE, and any tool steps the agent took are
// shown alongside the answer. A guardrail refusal arrives as a single "blocked"
// turn. The conversation id returned on the first frame is threaded back into
// later turns so the server resumes context.

import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  deleteConversation,
  listConversations,
  listMessages,
  resumeChat,
  streamChat,
} from "../api/client";
import type {
  AskPagesReadInterrupt,
  ChatInterrupt,
  ChatMediaPart,
  Conversation,
  ChatMessage,
  PageRangeConfirmInterrupt,
  ResumeRequestBody,
  SpoilerWarningInterrupt,
  ToolApprovalInterrupt,
  ToolStep,
} from "../api/types";
import { Alert, Button, EmptyState, Input, Textarea } from "./ui";

/** A just-sent attachment, rendered as a real thumbnail/player from the in-memory file. */
interface AttachmentPreview {
  kind: "image" | "audio";
  name: string;
  url: string;
}

/** An attachment on a reloaded historical message: no bytes, just a count by kind. */
interface AttachmentBadge {
  kind: "image" | "audio";
  count: number;
}

/** One rendered turn in the transcript (user prompt or assistant reply). */
interface TurnView {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolSteps?: ToolStep[];
  blocked?: boolean;
  streaming?: boolean;
  /** Live progress description (e.g. "Planning how to respond..."), shown
   * only while streaming with no content yet — see `onNodeStatus` in `send`. */
  status?: string;
  previews?: AttachmentPreview[];
  badges?: AttachmentBadge[];
}

// Matches the backend's persisted attachment note (e.g. "[1 image attachment]" or
// "[2 image attachments, 1 audio attachment]"), the only trace of an attachment
// left once a conversation is reloaded — the original bytes are never persisted
// on the message itself.
const ATTACHMENT_NOTE_RE =
  /\n*\[(\d+ (?:image|audio) attachments?(?:, \d+ (?:image|audio) attachments?)*)\]$/;

/** Split a persisted message into its text and any attachment-note badges. */
function parseAttachmentNote(content: string): { text: string; badges: AttachmentBadge[] } {
  const match = content.match(ATTACHMENT_NOTE_RE);
  if (!match) return { text: content, badges: [] };
  const badges = match[1].split(", ").map((part) => {
    const [count, kind] = part.split(" ");
    return { kind: kind as "image" | "audio", count: Number(count) };
  });
  return { text: content.slice(0, match.index).trimEnd(), badges };
}

let turnCounter = 0;
const nextTurnId = (): string => `turn-${++turnCounter}`;

/** Read a file into a `ChatMediaPart` (base64, no data-URI prefix). */
function fileToPart(file: File): Promise<ChatMediaPart> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the attachment."));
    reader.onload = () => {
      const result = reader.result as string; // "data:<mime>;base64,<payload>"
      const base64 = result.slice(result.indexOf(",") + 1);
      const kind = file.type.startsWith("audio/") ? "audio" : "image";
      resolve({ kind, mime_type: file.type, data: base64 });
    };
    reader.readAsDataURL(file);
  });
}

// Auto-stop a live recording after this long, so a forgotten "Stop" click can't
// grow an attachment without bound (there's no other client-side size cap here,
// matching file attachments — the server's CHAT_MEDIA_MAX_BYTES is the backstop).
const MAX_RECORDING_SECONDS = 120;

/** True only when the browser can actually record microphone audio. */
const recordingSupported =
  typeof navigator !== "undefined" &&
  typeof MediaRecorder !== "undefined" &&
  !!navigator.mediaDevices?.getUserMedia;

/** The first of the backend's allowlisted audio mime types this browser can record. */
function pickRecordingMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return undefined;
  return ["audio/webm", "audio/mp4", "audio/ogg"].find((candidate) =>
    MediaRecorder.isTypeSupported(candidate),
  );
}

/** Render a duration in seconds as "m:ss" for the live recording timer. */
function formatSeconds(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/** Map a persisted transcript message onto a rendered turn (skips non-visible roles). */
function toTurn(message: ChatMessage): TurnView | null {
  if (message.role !== "user" && message.role !== "assistant") return null;
  const { text, badges } = parseAttachmentNote(message.content);
  return {
    id: message.id,
    role: message.role,
    content: text,
    toolSteps: message.tool_calls?.steps,
    badges: badges.length > 0 ? badges : undefined,
  };
}

export function Chat(): React.JSX.Element {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [turns, setTurns] = useState<TurnView[]>([]);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A turn paused for the user's approval/answer (HITL): which conversation and
  // assistant turn it belongs to, so resolving it can update the right bubble.
  const [pendingInterrupt, setPendingInterrupt] = useState<ChatInterrupt | null>(null);
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null);
  const [pendingTurnId, setPendingTurnId] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<number | null>(null);
  const recordStopTimeoutRef = useRef<number | null>(null);
  // Object URLs backing sent-attachment previews, revoked on unmount so they
  // don't outlive the images/players that reference them.
  const previewUrlsRef = useRef<string[]>([]);

  const clearRecordingTimers = useCallback(() => {
    if (recordTimerRef.current !== null) {
      window.clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
    if (recordStopTimeoutRef.current !== null) {
      window.clearTimeout(recordStopTimeoutRef.current);
      recordStopTimeoutRef.current = null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access was denied or is unavailable.");
      return;
    }
    const mimeType = pickRecordingMimeType();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    recordedChunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      clearRecordingTimers();
      // recorder.mimeType often reports a codec-qualified string (e.g.
      // "audio/webm;codecs=opus"), but the backend's allowlist matches the
      // bare mime type exactly — strip any ";codecs=..." parameter, and prefer
      // the type we explicitly requested over whatever the browser echoes back.
      const rawType = mimeType || recorder.mimeType || "audio/webm";
      const blobType = rawType.split(";")[0];
      const blob = new Blob(recordedChunksRef.current, { type: blobType });
      const extension = blobType.includes("mp4") ? "m4a" : blobType.includes("ogg") ? "ogg" : "webm";
      const file = new File([blob], `voice-note-${Date.now()}.${extension}`, { type: blobType });
      setAttachments((prev) => [...prev, file]);
      setRecording(false);
      setRecordSeconds(0);
    };
    mediaRecorderRef.current = recorder;
    recorder.start();
    setRecording(true);
    setRecordSeconds(0);
    recordTimerRef.current = window.setInterval(() => setRecordSeconds((s) => s + 1), 1000);
    recordStopTimeoutRef.current = window.setTimeout(() => {
      if (recorder.state !== "inactive") recorder.stop();
    }, MAX_RECORDING_SECONDS * 1000);
  }, [clearRecordingTimers]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state !== "inactive") mediaRecorderRef.current?.stop();
  }, []);

  // Stop any in-progress recording (releasing the microphone) and revoke any
  // preview object URLs if the user navigates away, rather than leaking them.
  useEffect(() => {
    return () => {
      clearRecordingTimers();
      if (mediaRecorderRef.current?.state !== "inactive") mediaRecorderRef.current?.stop();
      previewUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [clearRecordingTimers]);

  const refreshConversations = useCallback(async () => {
    const page = await listConversations();
    setConversations(page.items);
  }, []);

  useEffect(() => {
    refreshConversations().catch(() => setError("Couldn't load your conversations."));
  }, [refreshConversations]);

  const openConversation = useCallback(async (id: string) => {
    setError(null);
    setActiveId(id);
    setPendingInterrupt(null);
    try {
      const page = await listMessages(id);
      setTurns(page.items.map(toTurn).filter((t): t is TurnView => t !== null));
    } catch {
      setError("Couldn't load that conversation.");
    }
  }, []);

  const startNewConversation = useCallback(() => {
    setActiveId(null);
    setTurns([]);
    setError(null);
    setPendingInterrupt(null);
  }, []);

  const removeConversation = useCallback(
    async (id: string) => {
      setError(null);
      try {
        await deleteConversation(id);
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (id === activeId) startNewConversation();
      } catch {
        setError("Couldn't delete that conversation.");
      }
    },
    [activeId, startNewConversation],
  );

  // Update a single turn in place (used to grow the streaming assistant reply).
  const patchTurn = useCallback((id: string, patch: Partial<TurnView>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if ((!text && attachments.length === 0) || busy) return;
    setBusy(true);
    setError(null);

    let parts: ChatMediaPart[];
    try {
      parts = await Promise.all(attachments.map(fileToPart));
    } catch {
      setError("Couldn't read an attachment.");
      setBusy(false);
      return;
    }

    const previews: AttachmentPreview[] = attachments.map((f) => {
      const url = URL.createObjectURL(f);
      previewUrlsRef.current.push(url);
      return { kind: f.type.startsWith("audio/") ? "audio" : "image", name: f.name, url };
    });
    const userTurn: TurnView = {
      id: nextTurnId(),
      role: "user",
      content: text,
      previews: previews.length > 0 ? previews : undefined,
    };
    const assistantId = nextTurnId();
    const assistantTurn: TurnView = {
      id: assistantId,
      role: "assistant",
      content: "",
      toolSteps: [],
      streaming: true,
    };
    setTurns((prev) => [...prev, userTurn, assistantTurn]);
    setInput("");
    setAttachments([]);

    let createdId: string | null = null;
    try {
      await streamChat(
        {
          message: text,
          parts: parts.length > 0 ? parts : undefined,
          conversation_id: activeId,
        },
        {
          onConversation: (id) => {
            createdId = id;
            if (!activeId) setActiveId(id);
          },
          onToolCall: (step) =>
            patchTurnAppendTool(assistantId, {
              name: step.name,
              args: step.args,
              result: "",
            }),
          onToolResult: (step) => patchTurnToolResult(assistantId, step.name, step.content),
          onNodeStatus: (status) => patchTurn(assistantId, { status: status.description }),
          onToken: (chunk) =>
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantId ? { ...t, content: t.content + chunk } : t,
              ),
            ),
          onDone: (answer) => patchTurn(assistantId, { content: answer, streaming: false }),
          onBlocked: (reason) =>
            patchTurn(assistantId, { content: reason, blocked: true, streaming: false }),
          onInterrupt: (interrupt) => {
            // page_range_confirm pauses after generate, so recap tokens may
            // already be in the bubble — keep them. Other HITL kinds pause
            // before an answer exists.
            patchTurn(assistantId, {
              ...(interrupt.kind === "page_range_confirm"
                ? {}
                : { content: "Waiting for your input…" }),
              streaming: false,
            });
            setPendingTurnId(assistantId);
            setPendingConversationId(createdId ?? activeId);
            setPendingInterrupt(interrupt);
          },
        },
      );
    } catch {
      patchTurn(assistantId, {
        content: "Something went wrong. Please try again.",
        blocked: true,
        streaming: false,
      });
    } finally {
      setBusy(false);
      // A brand-new conversation now exists server-side; refresh the picker so its
      // (freshly-titled) entry shows up.
      if (!activeId && createdId) refreshConversations().catch(() => undefined);
    }
  }, [input, attachments, busy, activeId, patchTurn, refreshConversations]);

  // Submit the user's decision/answer for the pending prompt. A resume may
  // itself pause again (another gated step on the same turn) — in that case
  // the prompt is replaced rather than cleared, and the bubble stays pending.
  const resolveInterrupt = useCallback(
    async (body: ResumeRequestBody) => {
      if (!pendingConversationId || !pendingTurnId) return;
      setResuming(true);
      setError(null);
      try {
        const result = await resumeChat(pendingConversationId, body);
        if (result.interrupt) {
          setPendingInterrupt(result.interrupt);
        } else {
          setPendingInterrupt(null);
          setPendingTurnId(null);
          setPendingConversationId(null);
          patchTurn(pendingTurnId, {
            content: result.answer,
            toolSteps: result.tool_steps,
            streaming: false,
          });
        }
      } catch {
        setError("Couldn't submit your response. Please try again.");
      } finally {
        setResuming(false);
      }
    },
    [pendingConversationId, pendingTurnId, patchTurn],
  );

  // Append a tool step to the streaming assistant turn (a call, result pending).
  function patchTurnAppendTool(id: string, step: ToolStep): void {
    setTurns((prev) =>
      prev.map((t) => (t.id === id ? { ...t, toolSteps: [...(t.toolSteps ?? []), step] } : t)),
    );
  }

  // Fill in the result of the most recent matching tool call.
  function patchTurnToolResult(id: string, name: string, result: string): void {
    setTurns((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const steps = [...(t.toolSteps ?? [])];
        for (let i = steps.length - 1; i >= 0; i--) {
          if (steps[i].name === name && steps[i].result === "") {
            steps[i] = { ...steps[i], result };
            break;
          }
        }
        return { ...t, toolSteps: steps };
      }),
    );
  }

  const onFiles = (files: FileList | null) => {
    if (!files) return;
    setAttachments(Array.from(files));
  };

  return (
    <section aria-labelledby="chat-heading" className="mx-auto flex max-w-3xl flex-col">
      <h2 id="chat-heading" className="text-xl font-semibold text-stone-900">
        Ask your reading assistant
      </h2>

      <div className="mt-3 flex items-center gap-2 overflow-x-auto pb-2">
        <Button variant="secondary" size="sm" className="shrink-0" onClick={startNewConversation}>
          + New conversation
        </Button>
        <ul aria-label="Your conversations" className="flex gap-2">
          {conversations.map((c) => (
            <li key={c.id} className="shrink-0">
              <div
                className={clsx(
                  "flex items-center gap-1 rounded-full py-1 pl-3 pr-1.5 transition-colors",
                  c.id === activeId
                    ? "bg-indigo-100 text-indigo-700"
                    : "bg-stone-100 text-stone-600 hover:bg-stone-200",
                )}
              >
                <button
                  type="button"
                  aria-current={c.id === activeId}
                  onClick={() => void openConversation(c.id)}
                  className="text-sm font-medium"
                >
                  {c.title || "Untitled conversation"}
                </button>
                <button
                  type="button"
                  aria-label="Delete conversation"
                  onClick={() => void removeConversation(c.id)}
                  className="rounded-full px-1 text-stone-400 hover:bg-stone-200 hover:text-stone-700"
                >
                  ×
                </button>
              </div>
            </li>
          ))}
        </ul>
      </div>

      {error && <Alert className="mb-3">{error}</Alert>}

      <ol
        aria-label="Conversation"
        data-testid="transcript"
        className="min-h-[16rem] flex-1 space-y-3 rounded-xl border border-stone-200 bg-white p-4"
      >
        {turns.length === 0 && (
          <EmptyState>Ask a question about your books to get started.</EmptyState>
        )}
        {turns.map((turn) => (
          <TurnItem key={turn.id} turn={turn} />
        ))}
      </ol>

      {pendingInterrupt && (
        <div className="mt-3">
          <InterruptPrompt
            interrupt={pendingInterrupt}
            busy={resuming}
            onSubmit={(body) => void resolveInterrupt(body)}
          />
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
        className="mt-3 space-y-2 rounded-xl border border-stone-200 bg-white p-3"
      >
        <label htmlFor="chat-input" className="sr-only">
          Message
        </label>
        <Textarea
          id="chat-input"
          value={input}
          disabled={busy || !!pendingInterrupt}
          placeholder="Ask about your books…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          rows={2}
          className="border-none px-1 py-0 focus-visible:outline-none"
        />

        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <label
              htmlFor="chat-attach"
              className="cursor-pointer text-xs font-medium text-stone-500 underline hover:text-stone-700"
            >
              Attach audio or image
            </label>
            <Input
              id="chat-attach"
              type="file"
              accept="audio/*,image/*"
              multiple
              disabled={busy || !!pendingInterrupt || recording}
              onChange={(e) => onFiles(e.target.files)}
              className="hidden"
            />
            {recordingSupported && (
              <Button
                type="button"
                variant={recording ? "danger" : "secondary"}
                size="sm"
                disabled={busy || !!pendingInterrupt}
                onClick={() => (recording ? stopRecording() : void startRecording())}
              >
                {recording ? `⏹ Stop (${formatSeconds(recordSeconds)})` : "🎙 Record"}
              </Button>
            )}
            {attachments.length > 0 && (
              <span data-testid="attachment-count" className="text-xs text-stone-500">
                {attachments.length} attached
              </span>
            )}
          </div>

          <Button
            type="submit"
            variant="primary"
            size="sm"
            disabled={busy || !!pendingInterrupt || recording}
          >
            {busy ? "Sending…" : "Send"}
          </Button>
        </div>
      </form>
    </section>
  );
}

/** One transcript entry: the author, the text, and any tool steps taken. */
function TurnItem({ turn }: { turn: TurnView }): React.JSX.Element {
  const isUser = turn.role === "user";
  return (
    <li data-testid={`turn-${turn.role}`} className={clsx("flex", isUser && "justify-end")}>
      <div className={clsx("max-w-[85%] space-y-1", isUser && "text-right")}>
        <span className="text-xs font-medium text-stone-400">{isUser ? "You" : "Assistant"}</span>
        {(turn.content || turn.streaming) && (
          <div
            className={clsx(
              "rounded-2xl px-3.5 py-2 text-sm",
              isUser && "bg-indigo-600 text-white",
              !isUser && !turn.blocked && "bg-stone-100 text-stone-900",
              turn.blocked && "bg-red-50 text-red-800",
            )}
          >
            <span role={turn.blocked ? "alert" : undefined}>
              {turn.content}
              {turn.streaming && !turn.content && <em> {turn.status ?? "…"}</em>}
            </span>
          </div>
        )}
        {turn.previews && turn.previews.length > 0 && (
          <div
            data-testid="attachment-previews"
            className={clsx("flex flex-wrap gap-2", isUser && "justify-end")}
          >
            {turn.previews.map((preview, i) =>
              preview.kind === "image" ? (
                <img
                  key={i}
                  src={preview.url}
                  alt={preview.name}
                  className="h-20 w-20 rounded-lg border border-stone-200 object-cover"
                />
              ) : (
                <audio key={i} controls src={preview.url} className="h-8 max-w-[220px]" />
              ),
            )}
          </div>
        )}
        {turn.badges && turn.badges.length > 0 && (
          <div
            data-testid="attachment-badges"
            className={clsx("flex flex-wrap gap-1.5", isUser && "justify-end")}
          >
            {turn.badges.map((badge, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-0.5 text-xs text-stone-600"
              >
                <span aria-hidden="true">{badge.kind === "image" ? "🖼️" : "🎤"}</span>
                {badge.count} {badge.kind}
                {badge.count > 1 ? "s" : ""}
              </span>
            ))}
          </div>
        )}
        {turn.toolSteps && turn.toolSteps.length > 0 && (
          <details className="text-left text-xs text-stone-500">
            <summary className="cursor-pointer">{turn.toolSteps.length} tool step(s)</summary>
            <ul className="mt-1 space-y-0.5 pl-3">
              {turn.toolSteps.map((step, i) => (
                <li key={i}>
                  <code className="rounded bg-stone-100 px-1 py-0.5">{step.name}</code>
                  {step.result && <span>: {step.result}</span>}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </li>
  );
}

/** Renders the pending HITL prompt, dispatching to the shape matching its `kind`. */
function InterruptPrompt({
  interrupt,
  busy,
  onSubmit,
}: {
  interrupt: ChatInterrupt;
  busy: boolean;
  onSubmit: (body: ResumeRequestBody) => void;
}): React.JSX.Element {
  switch (interrupt.kind) {
    case "tool_approval":
      return <ToolApprovalPrompt interrupt={interrupt} busy={busy} onSubmit={onSubmit} />;
    case "page_range_confirm":
      return <PageRangeConfirmPrompt interrupt={interrupt} busy={busy} onSubmit={onSubmit} />;
    case "ask_pages_read":
      return <AskPagesReadPrompt interrupt={interrupt} busy={busy} onSubmit={onSubmit} />;
    case "spoiler_warning":
      return <SpoilerWarningPrompt interrupt={interrupt} busy={busy} onSubmit={onSubmit} />;
  }
}

/** A gated tool call awaiting approve/deny/edit before it runs. */
function ToolApprovalPrompt({
  interrupt,
  busy,
  onSubmit,
}: {
  interrupt: ToolApprovalInterrupt;
  busy: boolean;
  onSubmit: (body: ResumeRequestBody) => void;
}): React.JSX.Element {
  const [editing, setEditing] = useState(false);
  const [argsText, setArgsText] = useState(() =>
    JSON.stringify(interrupt.tool_call.args, null, 2),
  );
  const [argsError, setArgsError] = useState<string | null>(null);

  const submitEdit = () => {
    try {
      const args = JSON.parse(argsText) as Record<string, unknown>;
      setArgsError(null);
      onSubmit({ decision: "edit", args });
    } catch {
      setArgsError("That's not valid JSON.");
    }
  };

  return (
    <div
      aria-label="Approval needed"
      data-testid="interrupt-tool-approval"
      className="space-y-3 rounded-xl border border-amber-200 bg-amber-50 p-4"
    >
      <p role="alert" className="text-sm text-amber-800">
        {interrupt.reason}
      </p>
      <p>
        <code className="rounded bg-white px-1.5 py-0.5 text-sm text-stone-700">
          {interrupt.tool_call.name}
        </code>
      </p>
      {editing ? (
        <>
          <label htmlFor="interrupt-args" className="block text-sm font-medium text-stone-700">
            Edit arguments (JSON)
          </label>
          <Textarea
            id="interrupt-args"
            value={argsText}
            onChange={(e) => setArgsText(e.target.value)}
            rows={4}
            className="font-mono text-xs"
          />
          {argsError && <Alert>{argsError}</Alert>}
          <Button variant="primary" size="sm" disabled={busy} onClick={submitEdit}>
            Submit edit
          </Button>
        </>
      ) : (
        <div className="flex gap-2">
          <Button variant="primary" size="sm" disabled={busy} onClick={() => onSubmit({ decision: "approve" })}>
            Approve
          </Button>
          <Button variant="danger" size="sm" disabled={busy} onClick={() => onSubmit({ decision: "deny" })}>
            Deny
          </Button>
          <Button variant="secondary" size="sm" disabled={busy} onClick={() => setEditing(true)}>
            Edit
          </Button>
        </div>
      )}
    </div>
  );
}

/** Confirm (or edit) the page range before it's saved as a summary memory (FR-4.6). */
function PageRangeConfirmPrompt({
  interrupt,
  busy,
  onSubmit,
}: {
  interrupt: PageRangeConfirmInterrupt;
  busy: boolean;
  onSubmit: (body: ResumeRequestBody) => void;
}): React.JSX.Element {
  const [pageStart, setPageStart] = useState(interrupt.proposal.page_start);
  const [pageEnd, setPageEnd] = useState(interrupt.proposal.page_end);

  return (
    <div
      aria-label="Confirm summary range"
      data-testid="interrupt-page-range-confirm"
      className="space-y-3 rounded-xl border border-indigo-200 bg-indigo-50 p-4"
    >
      <p role="alert" className="text-sm text-indigo-800">
        Save a summary of <strong>{interrupt.document_title}</strong>, pages {pageStart}-
        {pageEnd}? {interrupt.proposal.proposal_reason}.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label htmlFor="range-start" className="block text-xs font-medium text-stone-600">
            From page
          </label>
          <Input
            id="range-start"
            type="number"
            min={1}
            value={pageStart}
            onChange={(e) => setPageStart(Number(e.target.value))}
            className="w-24"
          />
        </div>
        <div className="space-y-1">
          <label htmlFor="range-end" className="block text-xs font-medium text-stone-600">
            To page
          </label>
          <Input
            id="range-end"
            type="number"
            min={1}
            value={pageEnd}
            onChange={(e) => setPageEnd(Number(e.target.value))}
            className="w-24"
          />
        </div>
        <Button
          variant="primary"
          size="sm"
          disabled={busy}
          onClick={() => onSubmit({ decision: "edit", page_start: pageStart, page_end: pageEnd })}
        >
          Save summary
        </Button>
        <Button variant="secondary" size="sm" disabled={busy} onClick={() => onSubmit({ decision: "deny" })}>
          Don&rsquo;t save
        </Button>
      </div>
    </div>
  );
}

/** Ask which pages have been read when prior context is missing (FR-4.7). */
function AskPagesReadPrompt({
  interrupt,
  busy,
  onSubmit,
}: {
  interrupt: AskPagesReadInterrupt;
  busy: boolean;
  onSubmit: (body: ResumeRequestBody) => void;
}): React.JSX.Element {
  const [pageEnd, setPageEnd] = useState<number | "">("");

  return (
    <form
      aria-label="Which pages have you read"
      data-testid="interrupt-ask-pages-read"
      onSubmit={(e) => {
        e.preventDefault();
        if (pageEnd === "") return;
        onSubmit({ page_end: pageEnd });
      }}
      className="space-y-3 rounded-xl border border-indigo-200 bg-indigo-50 p-4"
    >
      <p role="alert" className="text-sm text-indigo-800">
        {interrupt.reason}
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label htmlFor="pages-read" className="block text-xs font-medium text-stone-600">
            Up to which page of <strong>{interrupt.document_title}</strong> have you read?
          </label>
          <Input
            id="pages-read"
            type="number"
            min={1}
            required
            value={pageEnd}
            onChange={(e) => setPageEnd(e.target.value === "" ? "" : Number(e.target.value))}
            className="w-24"
          />
        </div>
        <Button type="submit" variant="primary" size="sm" disabled={busy}>
          Submit
        </Button>
      </div>
    </form>
  );
}

/** Warn before revealing content past the reader's current page (FR-18.4). */
function SpoilerWarningPrompt({
  interrupt,
  busy,
  onSubmit,
}: {
  interrupt: SpoilerWarningInterrupt;
  busy: boolean;
  onSubmit: (body: ResumeRequestBody) => void;
}): React.JSX.Element {
  return (
    <div
      aria-label="Spoiler warning"
      data-testid="interrupt-spoiler-warning"
      className="space-y-3 rounded-xl border border-amber-200 bg-amber-50 p-4"
    >
      <p role="alert" className="text-sm text-amber-800">
        That answer goes beyond page {interrupt.current_page} of{" "}
        <strong>{interrupt.document_title}</strong>: {interrupt.reason}
      </p>
      <div className="flex gap-2">
        <Button variant="secondary" size="sm" disabled={busy} onClick={() => onSubmit({ decision: "approve" })}>
          Show me anyway
        </Button>
        <Button variant="primary" size="sm" disabled={busy} onClick={() => onSubmit({ decision: "deny" })}>
          Keep it hidden
        </Button>
      </div>
    </div>
  );
}
