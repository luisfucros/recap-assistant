import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AuthProvider } from "../auth/AuthContext";
import { json, mockFetch } from "../test/apiMock";
import { Dashboard } from "./Dashboard";

const USER = {
  id: "u1",
  email: "ada@example.com",
  display_name: "Ada",
  preferred_language: "en",
  spoiler_safe: true,
};

const EMPTY_LIBRARY = { items: [], total: 0, page: 1, page_size: 50 };
const EMPTY_MEMORIES = { items: [], total: 0, page: 1, page_size: 50 };
const EMPTY_RECOMMENDATIONS = { items: [] };
const EMPTY_PROGRESS = { reading: [], completed: [], cancelled: [] };
const EMPTY_ANALYTICS = {
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
};

/** The read-only routes the Dashboard's child views fetch on mount. */
const BASE_ROUTES = {
  "GET /documents": () => json(EMPTY_LIBRARY),
  "GET /progress": () => json(EMPTY_PROGRESS),
  "GET /analytics": () => json(EMPTY_ANALYTICS),
  "GET /memory": () => json(EMPTY_MEMORIES),
  "GET /recommendations": () => json(EMPTY_RECOMMENDATIONS),
};

afterEach(() => vi.restoreAllMocks());

function renderDashboard() {
  render(
    <AuthProvider>
      <Dashboard />
    </AuthProvider>,
  );
}

describe("Dashboard", () => {
  it("greets the signed-in user", async () => {
    mockFetch({ "GET /users/me": () => json(USER), ...BASE_ROUTES });
    renderDashboard();

    expect(await screen.findByText(/welcome back, ada/i)).toBeInTheDocument();
  });

  it("persists a language change via PATCH and reflects it", async () => {
    const patch = vi.fn((init: RequestInit | undefined) => {
      const body = JSON.parse(init?.body as string) as { preferred_language: string };
      return json({ ...USER, preferred_language: body.preferred_language });
    });
    mockFetch({
      "GET /users/me": () => json(USER),
      ...BASE_ROUTES,
      "PATCH /users/me": patch,
    });
    renderDashboard();

    const select = (await screen.findByLabelText(/language/i)) as HTMLSelectElement;
    expect(select.value).toBe("en");

    fireEvent.change(select, { target: { value: "es" } });

    await waitFor(() => expect(patch).toHaveBeenCalled());
    await waitFor(() => expect(select.value).toBe("es"));
  });

  it("toggles the global spoiler-safe setting via PATCH", async () => {
    const patch = vi.fn((init: RequestInit | undefined) => {
      const body = JSON.parse(init?.body as string) as { spoiler_safe: boolean };
      return json({ ...USER, spoiler_safe: body.spoiler_safe });
    });
    mockFetch({
      "GET /users/me": () => json(USER),
      ...BASE_ROUTES,
      "PATCH /users/me": patch,
    });
    renderDashboard();

    const toggle = (await screen.findByLabelText(/spoiler-safe/i)) as HTMLInputElement;
    expect(toggle.checked).toBe(true);

    fireEvent.click(toggle);

    await waitFor(() => expect(patch).toHaveBeenCalled());
    const body = JSON.parse(patch.mock.calls[0][0]?.body as string);
    expect(body).toEqual({ spoiler_safe: false });
    await waitFor(() => expect(toggle.checked).toBe(false));
  });
});
