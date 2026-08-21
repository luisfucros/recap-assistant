import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Recommendation } from "../api/types";
import { json, mockFetch } from "../test/apiMock";
import { Recommendations } from "./Recommendations";

afterEach(() => {
  vi.restoreAllMocks();
});

function recommendation(overrides: Partial<Recommendation> = {}): Recommendation {
  return {
    title: "The Iliad",
    reason: "Because you completed The Odyssey",
    document_id: "doc-1",
    author: "Homer",
    url: null,
    score: 0.8,
    ...overrides,
  };
}

describe("Recommendations", () => {
  it("lists recommendations with their explanation", async () => {
    mockFetch({
      "GET /recommendations": () => json({ items: [recommendation()] }),
    });
    render(<Recommendations />);

    expect(await screen.findByText("The Iliad")).toBeInTheDocument();
    expect(screen.getByText(/by Homer/)).toBeInTheDocument();
    expect(screen.getByText("Because you completed The Odyssey")).toBeInTheDocument();
  });

  it("renders a link for a web-sourced recommendation", async () => {
    mockFetch({
      "GET /recommendations": () =>
        json({
          items: [
            recommendation({
              title: "A Great Read",
              author: null,
              document_id: null,
              url: "http://example.com/book",
              reason: 'From a web search for "books like The Odyssey"',
            }),
          ],
        }),
    });
    render(<Recommendations />);

    const link = await screen.findByRole("link", { name: "http://example.com/book" });
    expect(link).toHaveAttribute("href", "http://example.com/book");
  });

  it("shows a message when there is nothing to recommend yet", async () => {
    mockFetch({ "GET /recommendations": () => json({ items: [] }) });
    render(<Recommendations />);
    expect(await screen.findByText(/no recommendations yet/i)).toBeInTheDocument();
  });

  it("shows an error when the request fails", async () => {
    mockFetch({
      "GET /recommendations": () => json({ detail: "boom", code: "ERROR" }, 500),
    });
    render(<Recommendations />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't load/i);
  });

  it("refreshes on demand", async () => {
    let items: Recommendation[] = [];
    mockFetch({ "GET /recommendations": () => json({ items }) });
    render(<Recommendations />);
    await screen.findByText(/no recommendations yet/i);

    items = [recommendation()];
    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(await screen.findByText("The Iliad")).toBeInTheDocument();
  });
});
