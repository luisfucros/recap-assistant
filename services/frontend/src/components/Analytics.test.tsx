import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AnalyticsSummary } from "../api/types";
import { json, mockFetch } from "../test/apiMock";
import { Analytics } from "./Analytics";

afterEach(() => vi.restoreAllMocks());

function summary(overrides: Partial<AnalyticsSummary> = {}): AnalyticsSummary {
  return {
    window_days: 30,
    pages_read: 120,
    active_days: 6,
    pace_pages_per_day: 20,
    current_streak_days: 3,
    longest_streak_days: 5,
    documents_started: 2,
    documents_completed: 1,
    documents_cancelled: 0,
    pages_over_time: [
      { day: "2026-08-01", pages: 40 },
      { day: "2026-08-02", pages: 80 },
    ],
    ...overrides,
  };
}

describe("Analytics", () => {
  it("renders the headline reading metrics", async () => {
    mockFetch({ "GET /analytics": () => json(summary()) });
    render(<Analytics />);

    expect(await screen.findByText("120")).toBeInTheDocument(); // pages read
    expect(screen.getByText(/pace/i)).toBeInTheDocument();
    expect(screen.getByText("3d")).toBeInTheDocument(); // current streak
    expect(screen.getByText("5d")).toBeInTheDocument(); // longest streak
    // Pages-over-time rows render one entry per active day.
    expect(screen.getByText("2026-08-01")).toBeInTheDocument();
    expect(screen.getByText("2026-08-02")).toBeInTheDocument();
  });

  it("shows an empty state when nothing has been read", async () => {
    mockFetch({
      "GET /analytics": () => json(summary({ pages_read: 0, pages_over_time: [] })),
    });
    render(<Analytics />);

    expect(await screen.findByText(/no reading recorded yet/i)).toBeInTheDocument();
  });

  it("surfaces a load error", async () => {
    mockFetch({
      "GET /analytics": () =>
        json({ detail: "boom", code: "INTERNAL_ERROR" }, 500),
    });
    render(<Analytics />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't load your analytics/i);
  });

  it("refreshes on demand", async () => {
    let current = summary({ pages_read: 10 });
    mockFetch({ "GET /analytics": () => json(current) });
    render(<Analytics />);
    await screen.findByText("10");

    current = summary({ pages_read: 99 });
    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(await screen.findByText("99")).toBeInTheDocument();
  });
});
