import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { EvaluationRun } from "../api/types";
import { json, mockFetch } from "../test/apiMock";
import { Admin } from "./Admin";

afterEach(() => vi.restoreAllMocks());

function pendingRun(overrides: Partial<EvaluationRun> = {}): EvaluationRun {
  return {
    id: "run-1",
    dataset_name: "sample_v1",
    dataset_version: "v1",
    status: "pending",
    prompt_version: "generate@v5",
    llm_provider: "anthropic",
    llm_model: "claude-x",
    embedding_model: "text-embedding-3-small",
    results: {},
    summary: {},
    error: null,
    created_at: "2026-08-27T00:00:00Z",
    ...overrides,
  };
}

function completedRun(overrides: Partial<EvaluationRun> = {}): EvaluationRun {
  return pendingRun({
    status: "completed",
    summary: {
      cases: 3,
      retrieval: { hit_rate: 0.9, recall: 0.8, mrr: 0.7 },
      answer_quality: { faithfulness: 0.95, relevance: 0.85, citation_ok_rate: 1 },
      blocked: 0,
      interrupted: 0,
    },
    results: {
      cases: [
        {
          case_id: "q-opening",
          retrieval: { hit_rate: 1, recall: 1, mrr: 1 },
          answer: "The story opens on the river.",
          blocked: false,
          interrupted: false,
          answer_quality: {
            faithfulness: 1,
            relevance: 0.9,
            citation_ok: true,
            reasoning: "Grounded in the retrieved chunk.",
          },
        },
      ],
    },
    ...overrides,
  });
}

const DATASETS = { items: [{ name: "sample_v1", version: "v1" }] };

describe("Admin", () => {
  it("creates a user", async () => {
    const create = vi.fn(() =>
      json(
        {
          id: "u2",
          email: "new@example.com",
          display_name: "New",
          preferred_language: "en",
          spoiler_safe: true,
          is_admin: false,
        },
        201,
      ),
    );
    mockFetch({
      "GET /evaluations/datasets": () => json(DATASETS),
      "GET /evaluations": () => json({ items: [], total: 0, page: 1, page_size: 50 }),
      "POST /admin/users": create,
    });
    render(<Admin />);

    fireEvent.change(await screen.findByLabelText(/^email$/i), {
      target: { value: "new@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "hunter2!" } });
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => expect(create).toHaveBeenCalled());
    expect(await screen.findByRole("status")).toHaveTextContent(/created new@example.com/i);
  });

  it("surfaces a duplicate-email conflict", async () => {
    mockFetch({
      "GET /evaluations/datasets": () => json(DATASETS),
      "GET /evaluations": () => json({ items: [], total: 0, page: 1, page_size: 50 }),
      "POST /admin/users": () =>
        json({ detail: "taken", code: "USER_ALREADY_EXISTS" }, 409),
    });
    render(<Admin />);

    fireEvent.change(await screen.findByLabelText(/^email$/i), {
      target: { value: "ada@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "hunter2!" } });
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/already exists/i);
  });

  it("shows a pending run then a completed one after refresh", async () => {
    let items: EvaluationRun[] = [];
    const post = vi.fn(() => json(pendingRun(), 202));
    mockFetch({
      "GET /evaluations/datasets": () => json(DATASETS),
      "GET /evaluations": () => json({ items, total: items.length, page: 1, page_size: 50 }),
      "POST /evaluations/run": post,
    });
    render(<Admin />);

    expect(await screen.findByText(/no evaluation runs yet/i)).toBeInTheDocument();

    items = [pendingRun()];
    fireEvent.click(screen.getByRole("button", { name: /^run$/i }));

    expect(await screen.findByText(/queued/i)).toBeInTheDocument();
    await waitFor(() => expect(post).toHaveBeenCalled());

    items = [completedRun()];
    fireEvent.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(await screen.findByText(/completed/i)).toBeInTheDocument();
    expect(screen.getByText("3 cases")).toBeInTheDocument();
    expect(screen.getByText("90%")).toBeInTheDocument();
    expect(screen.getByText("95%")).toBeInTheDocument();
    expect(screen.getByText("Citations OK")).toBeInTheDocument();
    expect(screen.getByText("q-opening", { exact: false })).toBeInTheDocument();

    fireEvent.click(screen.getByText("q-opening", { exact: false }));
    expect(screen.getByText("OK")).toBeInTheDocument();
    expect(screen.getByText("The story opens on the river.")).toBeInTheDocument();
    expect(screen.getByText("Grounded in the retrieved chunk.")).toBeInTheDocument();
  });

  it("labels a blocked case instead of showing answer-quality scores", async () => {
    mockFetch({
      "GET /evaluations/datasets": () => json(DATASETS),
      "GET /evaluations": () =>
        json({
          items: [
            completedRun({
              summary: {
                cases: 1,
                retrieval: { hit_rate: 0, recall: 0, mrr: 0 },
                answer_quality: { faithfulness: 0, relevance: 0, citation_ok_rate: 0 },
                blocked: 1,
                interrupted: 0,
              },
              results: {
                cases: [
                  {
                    case_id: "off-topic",
                    retrieval: { hit_rate: 0, recall: 0, mrr: 0 },
                    answer: "",
                    blocked: true,
                    interrupted: false,
                    answer_quality: null,
                  },
                ],
              },
            }),
          ],
          total: 1,
          page: 1,
          page_size: 50,
        }),
    });
    render(<Admin />);

    expect(await screen.findByText("1 case")).toBeInTheDocument();
    expect(screen.getByText(/1 blocked/i)).toBeInTheDocument();
    expect(screen.getByText("Blocked by guardrail")).toBeInTheDocument();
  });
});
