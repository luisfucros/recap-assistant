import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { StoredMemory } from "../api/types";
import { json, mockFetch, noContent } from "../test/apiMock";
import { Memory } from "./Memory";

afterEach(() => {
  vi.restoreAllMocks();
});

function memory(overrides: Partial<StoredMemory> = {}): StoredMemory {
  return {
    id: "mem-1",
    type: "preference",
    content: "likes sci-fi",
    document_id: null,
    page_start: null,
    page_end: null,
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function memoryPage(items: StoredMemory[] = []) {
  return { items, total: items.length, page: 1, page_size: 50 };
}

describe("Memory", () => {
  it("lists stored memories, including a page-range summary", async () => {
    mockFetch({
      "GET /memory": () =>
        json(
          memoryPage([
            memory(),
            memory({
              id: "mem-2",
              type: "summary",
              content: "Odysseus leaves Troy.",
              document_id: "doc-1",
              page_start: 1,
              page_end: 20,
            }),
          ]),
        ),
    });
    render(<Memory />);

    expect(await screen.findByText("likes sci-fi")).toBeInTheDocument();
    expect(screen.getByText("Odysseus leaves Troy.")).toBeInTheDocument();
    expect(screen.getByText(/pp\. 1-20/)).toBeInTheDocument();
  });

  it("shows a message when nothing is saved yet", async () => {
    mockFetch({ "GET /memory": () => json(memoryPage()) });
    render(<Memory />);
    expect(await screen.findByText(/nothing saved yet/i)).toBeInTheDocument();
  });

  it("re-fetches filtered by kind when the filter changes", async () => {
    mockFetch({
      "GET /memory": () =>
        json(memoryPage([memory({ type: "habit", content: "reads at night" })])),
    });
    render(<Memory />);
    await screen.findByText("reads at night");

    fireEvent.change(screen.getByLabelText(/filter by kind/i), { target: { value: "habit" } });

    await waitFor(() => {
      const lastUrl = String(vi.mocked(fetch).mock.calls.at(-1)?.[0]);
      expect(lastUrl).toContain("type=habit");
    });
  });

  it("refreshes the list on demand", async () => {
    let memories: StoredMemory[] = [];
    mockFetch({ "GET /memory": () => json(memoryPage(memories)) });
    render(<Memory />);
    await screen.findByText(/nothing saved yet/i);

    memories = [memory({ content: "likes sci-fi" })];
    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(await screen.findByText("likes sci-fi")).toBeInTheDocument();
  });

  it("deletes a memory and refreshes the list", async () => {
    let deleted = false;
    mockFetch({
      "GET /memory": () =>
        json(memoryPage(deleted ? [] : [memory({ content: "likes sci-fi" })])),
      "DELETE /memory/mem-1": () => {
        deleted = true;
        return noContent();
      },
    });
    render(<Memory />);

    fireEvent.click(await screen.findByRole("button", { name: /delete/i }));

    await waitFor(() => expect(screen.getByText(/nothing saved yet/i)).toBeInTheDocument());
  });

  it("shows an error when deletion fails", async () => {
    mockFetch({
      "GET /memory": () => json(memoryPage([memory()])),
      "DELETE /memory/mem-1": () => json({ detail: "not found", code: "NOT_FOUND" }, 404),
    });
    render(<Memory />);

    fireEvent.click(await screen.findByRole("button", { name: /delete/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't delete/i);
  });
});
