// Thin fetch wrapper for the recap API.
//
// `credentials: "include"` sends the httpOnly auth cookies with every request
// (tokens are never read/stored in JS). In dev, Vite proxies /api to the API;
// in the container, nginx does. Override the base with VITE_API_BASE_URL.

import type {
  AnalyticsSummary,
  ChatInterrupt,
  ChatRequestBody,
  ChatResponse,
  ChatStreamHandlers,
  ConversationPage,
  DocumentPage,
  DocumentSummary,
  MemoryKind,
  MemoryPage,
  MessagePage,
  ProfileUpdate,
  ProgressUpdate,
  ReadingList,
  ReadingProgress,
  RecommendationsResponse,
  ResumeRequestBody,
  User,
  CreateUserRequest,
  EvaluationDataset,
  EvaluationRun,
  EvaluationRunPage,
} from "./types";

const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

/** An API error carrying the backend's stable `{ detail, code }` payload. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

// Paths that must never trigger a refresh-and-retry: refresh itself (avoids a
// loop) and the other pre-session auth routes, where a 401 is a normal
// "wrong credentials" / "not logged in yet" outcome, not an expired session.
const NO_REFRESH_PATHS = new Set(["/auth/login", "/auth/register", "/auth/refresh", "/auth/logout"]);

/** Invoked when a session can't be silently renewed (refresh cookie missing/expired). */
let sessionExpiredHandler: (() => void) | null = null;

/** Register the callback that clears client-side auth state on a hard session expiry. */
export function setSessionExpiredHandler(handler: (() => void) | null): void {
  sessionExpiredHandler = handler;
}

// Dedup concurrent refreshes: several always-mounted dashboard panels can all
// hit a 401 from the same expired access token within the same tick, and the
// refresh endpoint is rate-limited, so only one /auth/refresh call should ever
// be in flight at a time.
let refreshInFlight: Promise<boolean> | null = null;

function refreshSession(): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = fetch(`${API_BASE}/auth/refresh`, { method: "POST", credentials: "include" })
      .then((res) => res.ok)
      .catch(() => false)
      .finally(() => {
        refreshInFlight = null;
      });
  }
  return refreshInFlight;
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  // Let the browser set the multipart Content-Type (with boundary) for file
  // uploads; only default to JSON for regular bodies.
  const isFormData = init?.body instanceof FormData;
  const baseHeaders: Record<string, string> = isFormData
    ? {}
    : { "Content-Type": "application/json" };
  const doFetch = () =>
    fetch(`${API_BASE}${path}`, {
      credentials: "include",
      headers: { ...baseHeaders, ...(init?.headers ?? {}) },
      ...init,
    });

  const res = await doFetch();
  if (res.status !== 401 || NO_REFRESH_PATHS.has(path)) return res;

  // The access token (15 min TTL) likely just expired mid-session; the refresh
  // cookie lives far longer (7 days), so silently renew and retry once before
  // treating this as a real, user-visible auth failure.
  if (!(await refreshSession())) {
    sessionExpiredHandler?.();
    return res;
  }
  return doFetch();
}

export async function toApiError(res: Response): Promise<ApiError> {
  // Every backend error is `{ detail, code }`; fall back if the body isn't JSON.
  try {
    const body = (await res.json()) as { detail?: string; code?: string };
    return new ApiError(res.status, body.code ?? "ERROR", body.detail ?? res.statusText);
  } catch {
    return new ApiError(res.status, "ERROR", res.statusText);
  }
}

/** Issue a request, returning parsed JSON (or undefined for empty bodies). */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(path, init);
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Return true when the API health endpoint responds OK. */
export async function getHealth(): Promise<boolean> {
  try {
    return (await apiFetch("/health")).ok;
  } catch {
    return false;
  }
}

/** The current user, or null if the request isn't authenticated (401). */
export async function getMe(): Promise<User | null> {
  const res = await apiFetch("/users/me");
  if (res.status === 401) return null;
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as User;
}

export async function login(email: string, password: string): Promise<User> {
  return request<User>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function register(
  email: string,
  password: string,
  displayName?: string,
): Promise<User> {
  return request<User>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, display_name: displayName || null }),
  });
}

export async function logout(): Promise<void> {
  await request<void>("/auth/logout", { method: "POST" });
}

export async function updateProfile(patch: ProfileUpdate): Promise<User> {
  return request<User>("/users/me", { method: "PATCH", body: JSON.stringify(patch) });
}

/** Full-page navigation target that starts the Google OAuth flow. */
export function googleLoginUrl(): string {
  return `${API_BASE}/auth/google/login`;
}

/** A page of the current user's documents (newest first). */
export async function listDocuments(page = 1, pageSize = 50): Promise<DocumentPage> {
  return request<DocumentPage>(`/documents?page=${page}&page_size=${pageSize}`);
}

/**
 * Upload a document for ingestion. Throws `ApiError` with code
 * `DUPLICATE_DOCUMENT` (409), `UNSUPPORTED_MEDIA_TYPE` (415), or
 * `PAYLOAD_TOO_LARGE` (413) on the corresponding rejections.
 */
export async function uploadDocument(file: File): Promise<DocumentSummary> {
  const body = new FormData();
  body.append("file", file);
  return request<DocumentSummary>("/documents", { method: "POST", body });
}

/** Delete a document and all its data (chunks, vectors, stored original). */
export async function deleteDocument(id: string): Promise<void> {
  await request<void>(`/documents/${id}`, { method: "DELETE" });
}

/**
 * Re-queue a failed document's ingestion from scratch (the original upload is
 * reused, no re-upload needed). Throws `ApiError` with code
 * `DOCUMENT_NOT_FAILED` (409) if the document isn't currently `failed`.
 */
export async function retryDocument(id: string): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/documents/${id}/retry`, { method: "POST" });
}

/** The current user's tracked documents, grouped by reading status. */
export async function listProgress(): Promise<ReadingList> {
  return request<ReadingList>("/progress");
}

/** The current user's reading state for a document, or null if untracked (404). */
export async function getProgress(documentId: string): Promise<ReadingProgress | null> {
  const res = await apiFetch(`/progress/${documentId}`);
  if (res.status === 404) return null;
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as ReadingProgress;
}

/** Update reading state for a document (position, status, per-book spoiler override). */
export async function updateProgress(
  documentId: string,
  patch: ProgressUpdate,
): Promise<ReadingProgress> {
  return request<ReadingProgress>(`/progress/${documentId}`, {
    method: "PUT",
    body: JSON.stringify(patch),
  });
}

/** The current user's reading analytics over a trailing window (default 30 days). */
export async function getAnalytics(windowDays = 30): Promise<AnalyticsSummary> {
  return request<AnalyticsSummary>(`/analytics?window_days=${windowDays}`);
}

/** The current user's conversations, most-recently-active first (paginated). */
export async function listConversations(page = 1, pageSize = 50): Promise<ConversationPage> {
  return request<ConversationPage>(`/conversations?page=${page}&page_size=${pageSize}`);
}

/** A conversation's messages in chronological order (404 if not the caller's). */
export async function listMessages(
  conversationId: string,
  page = 1,
  pageSize = 100,
): Promise<MessagePage> {
  return request<MessagePage>(
    `/conversations/${conversationId}/messages?page=${page}&page_size=${pageSize}`,
  );
}

/** Delete a conversation, its messages, and its agent state (404 if not the caller's). */
export async function deleteConversation(conversationId: string): Promise<void> {
  await request<void>(`/conversations/${conversationId}`, { method: "DELETE" });
}

/**
 * Resume a turn paused for the user's approval/answer (HITL). May itself
 * return with `interrupt` set again if another gated step follows on the
 * same turn; otherwise `answer`/`tool_steps` carry the completed reply.
 */
export async function resumeChat(
  conversationId: string,
  body: ResumeRequestBody,
): Promise<ChatResponse> {
  return request<ChatResponse>(`/chat/${conversationId}/resume`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** A page of the current user's stored memories, newest first (optional kind filter). */
export async function listMemories(
  type?: MemoryKind,
  page = 1,
  pageSize = 50,
): Promise<MemoryPage> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  if (type) params.set("type", type);
  return request<MemoryPage>(`/memory?${params.toString()}`);
}

/** Delete a stored memory and its vector embedding. */
export async function deleteMemory(id: string): Promise<void> {
  await request<void>(`/memory/${id}`, { method: "DELETE" });
}

/** Explainable recommendations from the reader's own library (FR-5). */
export async function listRecommendations(limit = 5): Promise<RecommendationsResponse> {
  return request<RecommendationsResponse>(`/recommendations?limit=${limit}`);
}

/** Create a regular or admin account (admin-only). */
export async function createUser(body: CreateUserRequest): Promise<User> {
  return request<User>("/admin/users", { method: "POST", body: JSON.stringify(body) });
}

/** Shipped evaluation datasets (admin-only). */
export async function listEvaluationDatasets(): Promise<{ items: EvaluationDataset[] }> {
  return request<{ items: EvaluationDataset[] }>("/evaluations/datasets");
}

/** Evaluation runs, newest first (admin-only). */
export async function listEvaluations(page = 1, pageSize = 50): Promise<EvaluationRunPage> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  return request<EvaluationRunPage>(`/evaluations?${params.toString()}`);
}

/** Enqueue a dataset run; returns 202 pending (admin-only). */
export async function runEvaluation(datasetName: string): Promise<EvaluationRun> {
  return request<EvaluationRun>("/evaluations/run", {
    method: "POST",
    body: JSON.stringify({ dataset_name: datasetName }),
  });
}

/**
 * Send a chat turn and stream the answer as Server-Sent Events.
 *
 * The handlers fire in the backend's guaranteed order: a `conversation` frame
 * (the thread id), then any `tool_call`/`tool_result` pairs, then answer `token`
 * chunks, then a terminal `done` — or a lone `blocked` on a guardrail refusal.
 * `node_status` is the one exception: live progress that can arrive anywhere
 * before the terminal event, safe to ignore. Pass an `AbortSignal` to cancel an
 * in-flight turn. Throws `ApiError` if the request itself fails before
 * streaming starts.
 */
export async function streamChat(
  body: ChatRequestBody,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await apiFetch("/chat/stream", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await toApiError(res);
  await consumeSse(res, handlers);
}

/** Read an SSE response body frame-by-frame and dispatch to the handlers. */
async function consumeSse(res: Response, handlers: ChatStreamHandlers): Promise<void> {
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Frames are separated by a blank line; a partial frame stays buffered.
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      dispatchFrame(buffer.slice(0, boundary), handlers);
      buffer = buffer.slice(boundary + 2);
    }
  }
}

/** Parse one `event:`/`data:` SSE frame and invoke the matching handler. */
function dispatchFrame(frame: string, handlers: ChatStreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trim());
  }
  if (dataLines.length === 0) return;
  const data = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
  switch (event) {
    case "conversation":
      handlers.onConversation?.(data.conversation_id as string);
      break;
    case "tool_call":
      handlers.onToolCall?.(
        data as unknown as { name: string; args: Record<string, unknown>; id: string },
      );
      break;
    case "tool_result":
      handlers.onToolResult?.(data as unknown as { name: string; content: string; id: string });
      break;
    case "node_status":
      handlers.onNodeStatus?.(data as unknown as { node: string; description: string });
      break;
    case "token":
      handlers.onToken?.(data.text as string);
      break;
    case "done":
      handlers.onDone?.(data.answer as string, (data.trace_id as string | null) ?? null);
      break;
    case "blocked":
      handlers.onBlocked?.(data.reason as string);
      break;
    case "interrupt":
      handlers.onInterrupt?.(data as unknown as ChatInterrupt);
      break;
  }
}
