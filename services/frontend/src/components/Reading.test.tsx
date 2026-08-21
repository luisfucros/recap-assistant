import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DocumentSummary, ReadingProgress } from "../api/types";
import { json, mockFetch } from "../test/apiMock";
import { Reading } from "./Reading";

afterEach(() => vi.restoreAllMocks());

function doc(overrides: Partial<DocumentSummary> = {}): DocumentSummary {
  return {
    id: "d1",
    filename: "book.pdf",
    title: "The Book",
    author: null,
    format: "pdf",
    language: null,
    status: "indexed",
    failure_reason: null,
    page_count: 100,
    created_at: "2026-01-01T00:00:00Z",
    indexed_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function progressRow(overrides: Partial<ReadingProgress> = {}): ReadingProgress {
  return {
    document_id: "d1",
    current_page: 10,
    last_summarized_page: 0,
    status: "reading",
    spoiler_safe: null,
    last_accessed_at: "2026-01-02T00:00:00Z",
    ...overrides,
  };
}

const library = (items: DocumentSummary[]) => ({
  items,
  total: items.length,
  page: 1,
  page_size: 50,
});

const grouped = (reading: ReadingProgress[] = []) => ({
  reading,
  completed: [],
  cancelled: [],
});

describe("Reading", () => {
  it("shows an empty state when there are no documents", async () => {
    mockFetch({
      "GET /documents": () => json(library([])),
      "GET /progress": () => json(grouped()),
    });
    render(<Reading />);
    expect(await screen.findByText(/upload a document to start/i)).toBeInTheDocument();
  });

  it("renders a document with its current page and status", async () => {
    mockFetch({
      "GET /documents": () => json(library([doc()])),
      "GET /progress": () => json(grouped([progressRow({ current_page: 42 })])),
    });
    render(<Reading />);

    expect(await screen.findByText("The Book")).toBeInTheDocument();
    expect(screen.getByTestId("status-d1")).toHaveTextContent("Reading");
    const input = screen.getByLabelText(/current page/i) as HTMLInputElement;
    expect(input.value).toBe("42");
  });

  it("updates the current page via PUT", async () => {
    const put = vi.fn((init: RequestInit | undefined) => {
      const body = JSON.parse(init?.body as string) as { current_page: number };
      return json(progressRow({ current_page: body.current_page }));
    });
    mockFetch({
      "GET /documents": () => json(library([doc()])),
      "GET /progress": () => json(grouped([progressRow({ current_page: 10 })])),
      "PUT /progress/d1": put,
    });
    render(<Reading />);

    const input = (await screen.findByLabelText(/current page/i)) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "55" } });
    fireEvent.click(screen.getByRole("button", { name: /update/i }));

    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(JSON.parse(put.mock.calls[0][0]?.body as string)).toEqual({ current_page: 55 });
  });

  it("sets a per-book spoiler override via PUT", async () => {
    const put = vi.fn((_init: RequestInit | undefined) => json(progressRow({ spoiler_safe: false })));
    mockFetch({
      "GET /documents": () => json(library([doc()])),
      "GET /progress": () => json(grouped([progressRow()])),
      "PUT /progress/d1": put,
    });
    render(<Reading />);

    const select = (await screen.findByLabelText(/spoiler-safe for the book/i)) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "off" } });

    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(JSON.parse(put.mock.calls[0][0]?.body as string)).toEqual({ spoiler_safe: false });
  });

  it("refreshes on demand, picking up a document uploaded elsewhere", async () => {
    let docs: DocumentSummary[] = [];
    mockFetch({
      "GET /documents": () => json(library(docs)),
      "GET /progress": () => json(grouped()),
    });
    render(<Reading />);
    await screen.findByText(/upload a document to start/i);

    docs = [doc()];
    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(await screen.findByText("The Book")).toBeInTheDocument();
  });

  it("cancels a book via PUT", async () => {
    const put = vi.fn((_init: RequestInit | undefined) => json(progressRow({ status: "cancelled" })));
    mockFetch({
      "GET /documents": () => json(library([doc()])),
      "GET /progress": () => json(grouped([progressRow()])),
      "PUT /progress/d1": put,
    });
    render(<Reading />);

    fireEvent.click(await screen.findByRole("button", { name: /cancel/i }));

    await waitFor(() => expect(put).toHaveBeenCalled());
    expect(JSON.parse(put.mock.calls[0][0]?.body as string)).toEqual({ status: "cancelled" });
    await waitFor(() => expect(screen.getByTestId("status-d1")).toHaveTextContent("Cancelled"));
  });
});
