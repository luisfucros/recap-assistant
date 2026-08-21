import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DocumentSummary } from "../api/types";
import { json, mockFetch, noContent } from "../test/apiMock";
import { Library } from "./Library";

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

function doc(overrides: Partial<DocumentSummary> = {}): DocumentSummary {
  return {
    id: "d1",
    filename: "book.pdf",
    title: null,
    author: null,
    format: "pdf",
    language: null,
    status: "pending",
    failure_reason: null,
    page_count: null,
    created_at: "2026-01-01T00:00:00Z",
    indexed_at: null,
    ...overrides,
  };
}

function page(items: DocumentSummary[]) {
  return { items, total: items.length, page: 1, page_size: 50 };
}

function pdfFile(name = "book.pdf") {
  return new File([new Uint8Array([1, 2, 3])], name, { type: "application/pdf" });
}

describe("Library", () => {
  it("shows an empty state when there are no documents", async () => {
    mockFetch({ "GET /documents": () => json(page([])) });
    render(<Library />);
    expect(await screen.findByText(/no documents yet/i)).toBeInTheDocument();
  });

  it("uploads a PDF and shows it in the list", async () => {
    let docs: DocumentSummary[] = [];
    mockFetch({
      "GET /documents": () => json(page(docs)),
      "POST /documents": () => {
        docs = [doc({ status: "pending" })];
        return json(docs[0], 201);
      },
    });
    render(<Library />);
    await screen.findByText(/no documents yet/i);

    fireEvent.change(screen.getByLabelText(/upload a pdf/i), {
      target: { files: [pdfFile()] },
    });

    expect(await screen.findByText("book.pdf")).toBeInTheDocument();
    expect(screen.getByTestId("status-d1")).toHaveTextContent(/queued/i);
  });

  it("surfaces a friendly message when the document is a duplicate", async () => {
    mockFetch({
      "GET /documents": () => json(page([])),
      "POST /documents": () =>
        json({ detail: "already exists", code: "DUPLICATE_DOCUMENT" }, 409),
    });
    render(<Library />);
    await screen.findByText(/no documents yet/i);

    fireEvent.change(screen.getByLabelText(/upload a pdf/i), {
      target: { files: [pdfFile()] },
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(/already uploaded/i);
  });

  it("rejects a non-PDF with an explanatory message", async () => {
    mockFetch({
      "GET /documents": () => json(page([])),
      "POST /documents": () =>
        json({ detail: "unsupported", code: "UNSUPPORTED_MEDIA_TYPE" }, 415),
    });
    render(<Library />);
    await screen.findByText(/no documents yet/i);

    fireEvent.change(screen.getByLabelText(/upload a pdf/i), {
      target: { files: [pdfFile("notes.txt")] },
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(/only pdf/i);
  });

  it("deletes a document", async () => {
    let docs: DocumentSummary[] = [doc({ status: "indexed" })];
    mockFetch({
      "GET /documents": () => json(page(docs)),
      "DELETE /documents/d1": () => {
        docs = [];
        return noContent();
      },
    });
    render(<Library />);

    fireEvent.click(await screen.findByRole("button", { name: /delete/i }));

    await waitFor(() => expect(screen.queryByText("book.pdf")).not.toBeInTheDocument());
    expect(screen.getByText(/no documents yet/i)).toBeInTheDocument();
  });

  it("retries a failed document", async () => {
    let docs: DocumentSummary[] = [doc({ status: "failed", failure_reason: "parse failed" })];
    mockFetch({
      "GET /documents": () => json(page(docs)),
      "POST /documents/d1/retry": () => {
        docs = [doc({ status: "pending", failure_reason: null })];
        return json(docs[0]);
      },
    });
    render(<Library />);

    expect(await screen.findByText(/parse failed/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => expect(screen.getByTestId("status-d1")).toHaveTextContent(/queued/i));
    expect(screen.queryByText(/parse failed/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });

  it("surfaces a friendly message when retry fails", async () => {
    mockFetch({
      "GET /documents": () => json(page([doc({ status: "failed" })])),
      "POST /documents/d1/retry": () =>
        json({ detail: "not failed", code: "DOCUMENT_NOT_FAILED" }, 409),
    });
    render(<Library />);

    fireEvent.click(await screen.findByRole("button", { name: /retry/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't retry/i);
  });

  it("shows the filename alongside a title that doesn't match it", async () => {
    mockFetch({
      "GET /documents": () =>
        json(page([doc({ title: "0000320193-22-000070", filename: "apple-10k.pdf" })])),
    });
    render(<Library />);

    expect(await screen.findByText("0000320193-22-000070")).toBeInTheDocument();
    expect(screen.getByText("apple-10k.pdf")).toBeInTheDocument();
  });

  it("doesn't duplicate the name when there is no distinct title", async () => {
    mockFetch({ "GET /documents": () => json(page([doc({ title: null })])) });
    render(<Library />);

    expect(await screen.findAllByText("book.pdf")).toHaveLength(1);
  });

  it("refreshes the library on demand", async () => {
    let docs: DocumentSummary[] = [doc({ status: "processing" })];
    mockFetch({ "GET /documents": () => json(page(docs)) });
    render(<Library />);

    await screen.findByTestId("status-d1");
    docs = [doc({ status: "indexed" })];
    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    await waitFor(() => expect(screen.getByTestId("status-d1")).toHaveTextContent(/ready/i));
  });

  it("polls while a document is still ingesting and stops once it is ready", async () => {
    vi.useFakeTimers();
    let calls = 0;
    mockFetch({
      "GET /documents": () => {
        calls += 1;
        return json(page([doc({ status: calls === 1 ? "pending" : "indexed" })]));
      },
    });
    render(<Library />);

    // First load: pending → "Queued".
    await vi.waitFor(() => expect(screen.getByTestId("status-d1")).toHaveTextContent(/queued/i));

    // A poll interval later, the status refreshes to "Ready".
    await vi.advanceTimersByTimeAsync(3000);
    await vi.waitFor(() =>
      expect(screen.getByTestId("status-d1")).toHaveTextContent(/ready/i),
    );

    // Terminal now — no further polling happens.
    const callsAfterReady = calls;
    await vi.advanceTimersByTimeAsync(6000);
    expect(calls).toBe(callsAfterReady);
  });
});
