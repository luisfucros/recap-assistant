import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { json, mockFetch, unauthorized } from "./test/apiMock";

const USER = {
  id: "u1",
  email: "ada@example.com",
  display_name: "Ada",
  preferred_language: "en",
  spoiler_safe: true,
  is_admin: false,
};

/** The read-only routes the Dashboard's always-mounted child panels fetch. */
const BASE_ROUTES = {
  "GET /documents": () => json({ items: [], total: 0, page: 1, page_size: 50 }),
  "GET /progress": () => json({ reading: [], completed: [], cancelled: [] }),
  "GET /analytics": () =>
    json({
      window_days: 30,
      pages_read: 0,
      active_days: 0,
      pace_pages_per_day: 0,
      current_streak_days: 0,
      longest_streak_days: 0,
      documents_started: 0,
      documents_completed: 0,
      documents_cancelled: 0,
      pages_over_time: [],
    }),
  "GET /memory": () => json({ items: [], total: 0, page: 1, page_size: 50 }),
  "GET /recommendations": () => json({ items: [] }),
};

afterEach(() => vi.restoreAllMocks());

describe("App auth gate", () => {
  it("shows the login screen when the session probe is unauthenticated", async () => {
    mockFetch({ "GET /users/me": () => unauthorized() });

    render(<App />);

    expect(await screen.findByRole("heading", { name: /log in/i })).toBeInTheDocument();
  });

  it("shows the dashboard when a session is restored", async () => {
    mockFetch({ "GET /users/me": () => json(USER) });

    render(<App />);

    expect(await screen.findByText(/welcome back, ada/i)).toBeInTheDocument();
  });

  it("drops back to the login screen when a background request's session renewal fails", async () => {
    mockFetch({
      "GET /users/me": () => json(USER),
      ...BASE_ROUTES,
      // The access token expired mid-session and the refresh cookie is gone too.
      "GET /documents": () => unauthorized(),
      "POST /auth/refresh": () => unauthorized(),
    });

    render(<App />);
    await screen.findByText(/welcome back, ada/i);

    expect(await screen.findByRole("heading", { name: /log in/i })).toBeInTheDocument();
  });
});
