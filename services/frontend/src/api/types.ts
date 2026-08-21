// Shared API types, mirroring the backend's Pydantic models.

/** Supported languages (ISO 639-1) — mirrors the backend `Language` enum. */
export const LANGUAGES = ["en", "es", "de", "fr", "it"] as const;
export type Language = (typeof LANGUAGES)[number];

export const LANGUAGE_LABELS: Record<Language, string> = {
  en: "English",
  es: "Español",
  de: "Deutsch",
  fr: "Français",
  it: "Italiano",
};

/** Public view of a user (never carries the password hash). */
export interface User {
  id: string;
  email: string;
  display_name: string | null;
  preferred_language: Language;
  /** Global spoiler-safe default (FR-18); overridable per book/query. */
  spoiler_safe: boolean;
}

/** Fields a user may update on their own profile. */
export interface ProfileUpdate {
  display_name?: string | null;
  preferred_language?: Language;
  spoiler_safe?: boolean;
}

/** Ingestion lifecycle of a document — mirrors the backend `DocumentStatus`. */
export type DocumentStatus = "pending" | "processing" | "indexed" | "failed";

/** A status is terminal once ingestion has finished (success or failure). */
export function isTerminalStatus(status: DocumentStatus): boolean {
  return status === "indexed" || status === "failed";
}

/** Public view of a document — mirrors the backend `DocumentPublic`. */
export interface DocumentSummary {
  id: string;
  filename: string;
  title: string | null;
  author: string | null;
  format: string;
  language: Language | null;
  status: DocumentStatus;
  failure_reason: string | null;
  page_count: number | null;
  created_at: string;
  indexed_at: string | null;
}

/** A page of documents — mirrors the backend list envelope. */
export interface DocumentPage {
  items: DocumentSummary[];
  total: number;
  page: number;
  page_size: number;
}

/** Where a user stands in a document — mirrors the backend `ReadingStatus`. */
export type ReadingStatus = "not_started" | "reading" | "completed" | "cancelled";

export const READING_STATUS_LABELS: Record<ReadingStatus, string> = {
  not_started: "Not started",
  reading: "Reading",
  completed: "Completed",
  cancelled: "Cancelled",
};

/** A user's reading state for one document — mirrors `ReadingProgressPublic`. */
export interface ReadingProgress {
  document_id: string;
  current_page: number;
  last_summarized_page: number;
  status: ReadingStatus;
  /** Per-document spoiler-safe override; null = defer to the user default. */
  spoiler_safe: boolean | null;
  last_accessed_at: string;
}

/** The reading list grouped by status — mirrors `ReadingListResponse`. */
export interface ReadingList {
  reading: ReadingProgress[];
  completed: ReadingProgress[];
  cancelled: ReadingProgress[];
}

/** Fields updatable on a document's reading state (only provided ones change). */
export interface ProgressUpdate {
  current_page?: number;
  status?: ReadingStatus;
  /** null explicitly clears the per-document spoiler override. */
  spoiler_safe?: boolean | null;
}

/** Pages read on one calendar day — mirrors `PagesOnDay`. */
export interface PagesOnDay {
  day: string;
  pages: number;
}

/** A user's reading analytics — mirrors the backend `AnalyticsSummary`. */
export interface AnalyticsSummary {
  window_days: number;
  pages_read: number;
  active_days: number;
  pace_pages_per_day: number;
  current_streak_days: number;
  longest_streak_days: number;
  documents_started: number;
  documents_completed: number;
  documents_cancelled: number;
  pages_over_time: PagesOnDay[];
}

// --- Chat (agentic assistant) --------------------------------------------- //

/** A non-text chat attachment (FR-19), base64-encoded — mirrors `ChatMediaPart`. */
export interface ChatMediaPart {
  kind: "audio" | "image";
  mime_type: string;
  /** base64 of the raw bytes (no data-URI prefix). */
  data: string;
}

/** One turn's request body — mirrors the backend `ChatRequest`. */
export interface ChatRequestBody {
  message: string;
  parts?: ChatMediaPart[];
  /** Continue a thread, or omit/null to start a new one. */
  conversation_id?: string | null;
}

/** One tool the agent called this turn — mirrors `ToolStepPublic`. */
export interface ToolStep {
  name: string;
  args: Record<string, unknown>;
  result: string;
}

/** A chat thread — mirrors `ConversationPublic`. */
export interface Conversation {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

/** A page of conversations — mirrors the backend list envelope. */
export interface ConversationPage {
  items: Conversation[];
  total: number;
  page: number;
  page_size: number;
}

/** Who authored a persisted message — mirrors the backend `MessageRole`. */
export type MessageRole = "user" | "assistant" | "system" | "tool";

/** A persisted transcript message — mirrors `MessagePublic`. */
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  tool_calls: { steps: ToolStep[] } | null;
  created_at: string;
}

/** A page of a conversation's messages — mirrors the backend list envelope. */
export interface MessagePage {
  items: ChatMessage[];
  total: number;
  page: number;
  page_size: number;
}

/** A tool-approval interrupt (HITL) — mirrors the backend's `tool_approval` kind. */
export interface ToolApprovalInterrupt {
  kind: "tool_approval";
  tool_call: { name: string; args: Record<string, unknown>; id: string };
  reason: string;
}

/** A page-range-confirmation interrupt before saving a summary (FR-4.6). */
export interface PageRangeConfirmInterrupt {
  kind: "page_range_confirm";
  document_id: string;
  document_title: string;
  proposal: { page_start: number; page_end: number; proposal_reason: string };
}

/** An ask-which-pages-read interrupt when prior context is missing (FR-4.7). */
export interface AskPagesReadInterrupt {
  kind: "ask_pages_read";
  document_id: string;
  document_title: string;
  reason: string;
}

/** A generation-time spoiler warning before revealing ahead-of-position content (FR-18.4). */
export interface SpoilerWarningInterrupt {
  kind: "spoiler_warning";
  document_id: string;
  document_title: string;
  current_page: number;
  reason: string;
}

/** A turn paused for the user's decision/answer — mirrors the backend's interrupt payload. */
export type ChatInterrupt =
  | ToolApprovalInterrupt
  | PageRangeConfirmInterrupt
  | AskPagesReadInterrupt
  | SpoilerWarningInterrupt;

/** The body of `POST /chat/{id}/resume` — mirrors the backend `ResumeRequest`. */
export interface ResumeRequestBody {
  decision?: "approve" | "deny" | "edit";
  args?: Record<string, unknown>;
  page_start?: number;
  page_end?: number;
}

/**
 * The non-streamed result of a chat turn — mirrors the backend `ChatResponse`.
 * Also what `resumeChat` returns; a resumed turn may pause again (`interrupt`
 * set once more) if another gated step follows on the same turn.
 */
export interface ChatResponse {
  conversation_id: string;
  answer: string;
  blocked: boolean;
  tool_steps: ToolStep[];
  trace_id?: string | null;
  interrupt?: ChatInterrupt | null;
}

/** Callbacks for the ordered SSE events of a streamed chat turn. */
export interface ChatStreamHandlers {
  /** The thread id (first frame) — set it to continue the conversation. */
  onConversation?: (conversationId: string) => void;
  onToolCall?: (step: { name: string; args: Record<string, unknown>; id: string }) => void;
  onToolResult?: (step: { name: string; content: string; id: string }) => void;
  /**
   * Live progress: a node has started running, with a short description
   * (e.g. "Planning how to respond..."). Purely advisory — safe to ignore.
   */
  onNodeStatus?: (status: { node: string; description: string }) => void;
  /** A chunk of the answer as the model produces it. */
  onToken?: (text: string) => void;
  /**
   * The final, authoritative (sanitized) answer. `traceId` correlates the turn
   * to its trace when tracing is enabled (null/undefined otherwise).
   */
  onDone?: (answer: string, traceId?: string | null) => void;
  /** A guardrail refusal — the only event on a blocked turn. */
  onBlocked?: (reason: string) => void;
  /** The turn paused for the user's approval/answer (HITL); nothing else follows. */
  onInterrupt?: (interrupt: ChatInterrupt) => void;
}

// --- Long-term memory (FR-4.5 privacy view) ------------------------------- //

/** A kind of long-term memory — mirrors the backend `MemoryType` enum. */
export type MemoryKind = "preference" | "summary" | "concept" | "fact" | "habit" | "faq";

export const MEMORY_KIND_LABELS: Record<MemoryKind, string> = {
  preference: "Preference",
  summary: "Summary",
  concept: "Concept",
  fact: "Fact",
  habit: "Habit",
  faq: "FAQ",
};

/** A stored long-term memory — mirrors the backend `MemoryPublic`. */
export interface StoredMemory {
  id: string;
  type: MemoryKind;
  content: string;
  document_id: string | null;
  page_start: number | null;
  page_end: number | null;
  created_at: string;
}

/** A page of stored memories — mirrors the backend list envelope. */
export interface MemoryPage {
  items: StoredMemory[];
  total: number;
  page: number;
  page_size: number;
}

// --- Recommendations (FR-5) ------------------------------------------------ //

/** One explainable recommendation — mirrors the backend `RecommendationPublic`. */
export interface Recommendation {
  title: string;
  reason: string;
  /** Set only for a recommendation from the reader's own library. */
  document_id: string | null;
  author: string | null;
  /** Set only for a web-sourced recommendation. */
  url: string | null;
  score: number | null;
}

/** The `/recommendations` response — a bounded top-N, not a paginated list. */
export interface RecommendationsResponse {
  items: Recommendation[];
}
