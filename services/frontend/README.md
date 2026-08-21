# frontend

React single-page app (Vite + TypeScript), styled with Tailwind CSS v4. Talks to
the API under `/api` with `credentials: "include"` so httpOnly auth cookies ride
along — tokens are never read or stored in JavaScript. Built to a static bundle
and served by nginx, which reverse-proxies `/api` to the API service.

## Key entry points

- **`src/main.tsx`** — app bootstrap.
- **`src/App.tsx`** — root shell + the auth guard: shows the sign-in screen or the
  authenticated app based on `AuthContext`.
- **`src/auth/AuthContext.tsx`** — auth state + actions (login / register / logout /
  update profile). Probes `/users/me` on mount to restore a session (e.g. after
  the Google OAuth redirect). Never handles tokens — those stay in httpOnly cookies.
- **`src/components/AuthPage.tsx`** — email/password login+register with a
  **Continue with Google** link (a full-page navigation to `/api/v1/auth/google/login`).
- **`src/components/Dashboard.tsx`** — the signed-in shell: brand header, section
  nav (Chat/Library/Reading/Memory/Recommendations/Analytics — a sidebar on
  desktop, a horizontally-scrolling strip on mobile), account controls (language
  selector persisted via `PATCH /users/me`, spoiler-safe toggle, logout), and a
  focused content pane showing one section at a time. Every panel stays mounted
  (just visually hidden off-section) so switching sections never interrupts an
  in-flight poll or resets local state.
- **`src/components/ui/`** — the shared design-system primitives (`Button`,
  `Card`, `Input`/`Textarea`/`Select`/`Checkbox`/`FieldLabel`, `Badge`, `Alert`,
  `EmptyState`, `Spinner`) every panel composes instead of one-off markup. Each
  wraps its native element and forwards every prop untouched, so `htmlFor`/label
  associations and accessible roles are unaffected.
- **`src/index.css`** — the one global stylesheet: `@import "tailwindcss"` plus
  the app's base styles (background, focus ring). No `tailwind.config.js` —
  Tailwind v4 is CSS-first config.
- **`src/components/Library.tsx`** — upload a PDF (dropzone + file picker), see
  each document's ingestion status (polls while any is still processing), and
  delete. Maps `409/415/413` upload rejections to friendly messages.
- **`src/components/Chat.tsx`** — the reading assistant: a conversation picker and
  a live transcript. Turns stream token-by-token over Server-Sent Events (with the
  agent's tool steps surfaced and guardrail refusals shown), and a voice note or
  image can be attached (FR-19). Threads resume via the conversation id.
- **`src/api/client.ts`** — typed API client: `apiFetch` (cookie credentials,
  base URL), auth/document/progress/analytics calls, `streamChat` (the SSE parser
  for a chat turn), and `ApiError` carrying the backend's `{ detail, code }`.

## Authentication

Tokens live only in httpOnly cookies set by the API, so the SPA tracks *who* is
signed in (not any token). Google sign-in is a plain link: the browser navigates
to the API, which runs the OAuth flow and redirects back to `FRONTEND_URL` with
the cookies set; on load the app restores the session via `/users/me`. Routing is
a single conditional guard for now — URL routes arrive with more protected pages.

## Develop, test, build

```bash
npm install
npm run dev       # http://localhost:5173 (proxies /api → http://localhost:8000)
npm test          # Vitest + React Testing Library
npm run build     # type-check + production bundle to dist/
npm run lint      # tsc --noEmit
```

## Container

```bash
docker build -f services/frontend/Dockerfile -t recap-frontend .   # build context = repo root
```

Multi-stage: Node builds the bundle, nginx serves it (SPA fallback + `/api` proxy to the `api` service).
